"""Evaluate the report bot on multi-turn Atlas scenarios.

    python -m eval.atlas.run --split dev --cache .cache/llm            # live model
    python -m eval.atlas.run --split dev --oracle                      # harness self-check, no key

Metrics
  end-to-end   a wrong turn leaves wrong context for later turns (realistic, what users feel)
  teacher-forced
               every turn is asked with the GOLD history, isolating follow-up understanding from
               error propagation (diagnostic, not the headline number)
  per-scenario all turns correct
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from promptquery.bootstrap import ConfigError, Settings, build_engine, settings_from_env
from promptquery.conversation.state import PREVIEW_ROWS, Session, TurnRecord
from promptquery.engine import Engine
from promptquery.llm import LLMClient, LLMError

from .compare import compare_results
from .scenario import Scenario, Turn, load_scenarios


@dataclass
class TurnOutcome:
    scenario_id: str
    turn_idx: int  # 1-based
    user: str
    tags: list[str]
    mode: str  # e2e | forced
    expect: str
    status: str
    correct: bool
    reason: str = ""
    sql: str | None = None
    seconds: float = 0.0
    repaired: bool = False
    executed: bool = False


@dataclass
class Report:
    outcomes: list[TurnOutcome] = field(default_factory=list)

    def _acc(self, items: list[TurnOutcome]) -> float:
        return sum(o.correct for o in items) / len(items) if items else float("nan")

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for mode in ("e2e", "forced"):
            items = [o for o in self.outcomes if o.mode == mode]
            if not items:
                continue
            by_scn: dict[str, list[TurnOutcome]] = defaultdict(list)
            by_tag: dict[str, list[TurnOutcome]] = defaultdict(list)
            by_turn: dict[int, list[TurnOutcome]] = defaultdict(list)
            for o in items:
                by_scn[o.scenario_id].append(o)
                by_turn[o.turn_idx].append(o)
                for t in o.tags:
                    by_tag[t].append(o)
            answered = [o for o in items if o.expect == "ok"]
            refusals = [o for o in items if o.expect == "refused"]
            out[mode] = {
                "turns": len(items),
                "turn_accuracy": self._acc(items),
                "scenarios": len(by_scn),
                "scenario_accuracy": sum(all(o.correct for o in v) for v in by_scn.values()) / len(by_scn),
                "by_tag": {k: round(self._acc(v), 4) for k, v in sorted(by_tag.items())},
                "by_turn_index": {k: round(self._acc(v), 4) for k, v in sorted(by_turn.items())},
                "refusal_accuracy": self._acc(refusals),
                "sql_executed_rate": (sum(o.executed for o in answered) / len(answered)) if answered else None,
                "repair_rate": (sum(o.repaired for o in answered) / len(answered)) if answered else None,
                "mean_seconds": sum(o.seconds for o in items) / len(items),
            }
        return out


def _exec_gold(engine: Engine, sql: str) -> tuple[list[str], list[tuple]]:
    res = engine.db.run_scoped(sql, tenant_id=engine.tenant_id, max_rows=100_000, timeout_ms=30_000)
    return res.columns, res.rows


def _judge(engine: Engine, turn: Turn, result: Any) -> tuple[bool, str]:
    if turn.expect == "refused":
        return (result.status == "refused", f"status={result.status}")
    if result.status != "ok":
        return False, f"status={result.status}: {result.error or result.answer_text}"
    _, gold_rows = _exec_gold(engine, turn.gold_sql)
    cmp = compare_results(gold_rows, result.rows, turn.compare)
    return cmp.ok, cmp.reason


def _outcome(sc: Scenario, i: int, turn: Turn, mode: str, result: Any, ok: bool, why: str,
             seconds: float) -> TurnOutcome:
    notes = " ".join(result.notes)
    return TurnOutcome(
        scenario_id=sc.id, turn_idx=i, user=turn.user, tags=list(sc.tags), mode=mode,
        expect=turn.expect, status=result.status, correct=ok, reason=why, sql=result.sql,
        seconds=round(seconds, 2),
        repaired=("rejected attempt" in notes or "database error" in notes or "empty result" in notes
                  or "validator rejected" in notes),
        executed=result.status == "ok",
    )


def run_end_to_end(engine: Engine, sc: Scenario) -> list[TurnOutcome]:
    sid = f"e2e-{sc.id}"
    engine.reset_session(sid)
    outs = []
    for i, turn in enumerate(sc.turns, start=1):
        t0 = time.time()
        res = engine.ask(sid, turn.user)
        ok, why = _judge(engine, turn, res)
        outs.append(_outcome(sc, i, turn, "e2e", res, ok, why, time.time() - t0))
    return outs


def run_teacher_forced(engine: Engine, sc: Scenario) -> list[TurnOutcome]:
    """Ask each turn with gold history pre-loaded (no error propagation)."""
    outs = []
    history: list[TurnRecord] = []
    for i, turn in enumerate(sc.turns, start=1):
        sid = f"forced-{sc.id}-{i}"
        engine.reset_session(sid)
        engine.store._sessions[sid] = Session(id=sid, tenant_id=engine.tenant_id, turns=list(history))
        t0 = time.time()
        res = engine.ask(sid, turn.user)
        ok, why = _judge(engine, turn, res)
        outs.append(_outcome(sc, i, turn, "forced", res, ok, why, time.time() - t0))
        if turn.expect == "ok":
            cols, rows = _exec_gold(engine, turn.gold_sql)
            history.append(TurnRecord(
                turn_no=len(history) + 1, user_text=turn.user, standalone_question=turn.user,
                intent="NEW", status="ok", sql=turn.gold_sql, columns=cols,
                preview_rows=[tuple(r) for r in rows[:PREVIEW_ROWS]], row_count=len(rows),
            ))
    return outs


# ------------------------------------------------------------------ oracle / sabotage (no key)
class OracleLLM(LLMClient):
    """Returns the gold SQL for whatever question it is asked: validates the harness itself."""

    name = "oracle"
    model = "oracle"

    def __init__(self, scenarios: list[Scenario], sabotage: bool = False):
        self.gold: dict[str, Turn] = {}
        for sc in scenarios:
            for t in sc.turns:
                self.gold[t.user] = t
        self.sabotage = sabotage

    def generate(self, system: str, user: str) -> str:
        if "New message:" in user:  # the follow-up rewriter
            msg = user.rsplit("New message:", 1)[1].split("\n\nResolved reference", 1)[0].strip()
            return json.dumps({"intent": "NEW", "standalone_question": msg})
        return self.generate_messages(system, [{"role": "user", "content": user}])

    def generate_messages(self, system, messages, *, cache_system=False, max_tokens=None):
        text = messages[0]["content"]
        q = text.split("Question to answer now:", 1)[1].split("\n\n", 1)[0].strip()
        turn = self.gold[q]
        if turn.expect == "refused":
            return "REFUSE: not available"
        if self.sabotage:
            return "<plan>x</plan>\n```sql\nSELECT 1 AS wrong\n```"
        return f"<plan>oracle</plan>\n```sql\n{turn.gold_sql}\n```"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", default="dev", choices=["dev", "heldout", "all"])
    ap.add_argument("--ids", nargs="*", help="only these scenario ids")
    ap.add_argument("--mode", default="e2e", choices=["e2e", "forced", "both"])
    ap.add_argument("--oracle", action="store_true", help="use gold SQL as the model (harness check)")
    ap.add_argument("--sabotage", action="store_true", help="oracle that answers wrongly (checks failures count)")
    ap.add_argument("--cache", help="directory of the disk LLM cache")
    ap.add_argument("--refresh", action="store_true", help="ignore cached responses")
    ap.add_argument("--fail-under", type=float, help="exit 1 if turn accuracy is below this")
    ap.add_argument("--out", help="write the JSON report here")
    args = ap.parse_args(argv)

    scenarios = load_scenarios(split=None if args.split == "all" else args.split,
                               ids=set(args.ids or []))
    if not scenarios:
        print("no scenarios selected", file=sys.stderr)
        return 2

    try:
        base = settings_from_env()
    except ConfigError as e:
        print(f"configuration error: {e}", file=sys.stderr)
        return 2
    engines: dict[str, Engine] = {}

    def engine_for(tenant: str) -> Engine:
        if tenant not in engines:
            s = Settings(**{**asdict(base), "tenant": tenant})
            if args.oracle or args.sabotage:
                o = OracleLLM(scenarios, sabotage=args.sabotage)
                engines[tenant] = build_engine(s, llm=o, fast_llm=o)
            else:
                from promptquery.llm import AnthropicClient, CachingLLMClient

                llm: LLMClient = AnthropicClient(model=s.model)
                fast: LLMClient = AnthropicClient(model=s.fast_model)
                if args.cache:
                    llm = CachingLLMClient(llm, Path(args.cache), refresh=args.refresh)
                    fast = CachingLLMClient(fast, Path(args.cache), refresh=args.refresh)
                engines[tenant] = build_engine(s, llm=llm, fast_llm=fast)
        return engines[tenant]

    report = Report()
    for sc in scenarios:
        try:
            eng = engine_for(sc.tenant)
        except (LLMError, ConfigError) as e:
            print(f"cannot start the evaluation: {e}\n"
                  "(use --oracle to self-check the harness without a model)", file=sys.stderr)
            return 2
        if args.mode in ("e2e", "both"):
            report.outcomes += run_end_to_end(eng, sc)
        if args.mode in ("forced", "both"):
            report.outcomes += run_teacher_forced(eng, sc)

    summary = report.summary()
    print(json.dumps(summary, indent=2, default=str))
    failures = [o for o in report.outcomes if not o.correct]
    for o in failures[:25]:
        print(f"FAIL [{o.mode}] {o.scenario_id} turn {o.turn_idx}: {o.user!r} -> {o.reason}", file=sys.stderr)
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary,
                                              "outcomes": [asdict(o) for o in report.outcomes]},
                                             indent=2, default=str), encoding="utf-8")
    if args.fail_under is not None:
        key = "e2e" if "e2e" in summary else "forced"
        if summary[key]["turn_accuracy"] < args.fail_under:
            return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
