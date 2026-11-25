"""Multi-turn evaluation scenarios, authored as YAML.

    - id: fin_ar_01
      tenant: acme                 # Atlas tenant slug the conversation runs against
      split: dev                   # dev | heldout (heldout is only run at milestones)
      tags: [finance, ar]
      turns:
        - user: "Show open receivables by customer"
          gold_sql: "SELECT ..."   # executed through the same scoped, read-only path
          compare: {order: false}  # optional: order, extra_columns
        - user: "Show everyone's salary"
          expect: refused          # no gold SQL: the bot must decline
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .compare import CompareOptions

SCENARIO_DIR = Path(__file__).parent / "scenarios"


@dataclass(frozen=True)
class Turn:
    user: str
    gold_sql: str | None = None
    expect: str = "ok"  # ok | refused
    compare: CompareOptions = field(default_factory=CompareOptions)
    allow_empty: bool = False


@dataclass(frozen=True)
class Scenario:
    id: str
    tenant: str
    turns: tuple[Turn, ...]
    tags: tuple[str, ...] = ()
    split: str = "dev"

    @property
    def modules(self) -> tuple[str, ...]:
        return tuple(t for t in self.tags if t in MODULE_TAGS)


MODULE_TAGS = {
    "finance", "sales", "procurement", "inventory", "manufacturing", "hr", "crm", "projects",
    "quality", "maintenance", "hospitality", "safety",
}


def _parse_turn(raw: dict, scenario_id: str) -> Turn:
    expect = raw.get("expect", "ok")
    if expect not in ("ok", "refused"):
        raise ValueError(f"{scenario_id}: unknown expect {expect!r}")
    gold = raw.get("gold_sql")
    if expect == "ok" and not gold:
        raise ValueError(f"{scenario_id}: turn {raw.get('user')!r} needs gold_sql")
    c = raw.get("compare") or {}
    return Turn(
        user=raw["user"].strip(),
        gold_sql=gold.strip() if gold else None,
        expect=expect,
        compare=CompareOptions(order=bool(c.get("order", False)),
                               extra_columns=bool(c.get("extra_columns", True))),
        allow_empty=bool(raw.get("allow_empty", False)),
    )


def load_scenarios(
    directory: Path | None = None, split: str | None = None, ids: set[str] | None = None
) -> list[Scenario]:
    out: list[Scenario] = []
    seen: set[str] = set()
    for path in sorted((directory or SCENARIO_DIR).glob("*.yaml")):
        for raw in yaml.safe_load(path.read_text(encoding="utf-8")) or []:
            sid = raw["id"]
            if sid in seen:
                raise ValueError(f"duplicate scenario id {sid!r}")
            seen.add(sid)
            sc = Scenario(
                id=sid, tenant=raw.get("tenant", "acme"), split=raw.get("split", "dev"),
                tags=tuple(raw.get("tags", [])),
                turns=tuple(_parse_turn(t, sid) for t in raw["turns"]),
            )
            if len(sc.turns) > 6:
                raise ValueError(f"{sid}: more than 6 turns")
            if (split is None or sc.split == split) and (not ids or sid in ids):
                out.append(sc)
    return out
