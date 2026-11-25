"""The report-bot engine: one entry point, ``Engine.ask(session_id, text) -> TurnResult``.

CLI, HTTP API and evaluation all call this, so what is evaluated is exactly what ships.

Pipeline for a turn:
  turn-cap check -> follow-up rewrite -> generate (plan + SQL) -> as-of rewrite -> strict
  validation (+ tenant scoping, row cap) -> read-only tenant-scoped execution -> repair loop
  (validator rejection / DB error / suspicious empty result) -> summary -> audit.
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from .atlas.asof import apply_as_of, uses_wall_clock
from .atlas.examples import ExampleLibrary, render_examples
from .atlas.pack_loader import Pack
from .atlas.prompts import (
    build_system_prompt,
    build_user_message,
    parse_model_output,
    repair_message,
)
from .audit import AuditLog
from .conversation.context import format_history, format_rows, sanitize_cell
from .conversation.rewrite import rewrite_followup
from .conversation.state import MAX_TURNS, PREVIEW_ROWS, ConversationStore, Session, TurnRecord
from .db import QueryResult
from .llm import LLMClient
from .safety import UnsafeQuery, validate_report_query

TURN_LIMIT_MESSAGE = (
    f"This conversation has reached the limit of {MAX_TURNS} questions. "
    "Start a new session to continue."
)


@dataclass
class TurnResult:
    session_id: str
    turn_no: int
    status: str  # ok | refused | clarify | error | turn_limit
    answer_text: str = ""
    sql: str | None = None
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    standalone_question: str = ""
    intent: str = "NEW"
    notes: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_no": self.turn_no,
            "status": self.status,
            "answer_text": self.answer_text,
            "sql": self.sql,
            "columns": self.columns,
            "rows": [[_jsonable(v) for v in r] for r in self.rows],
            "row_count": self.row_count,
            "truncated": self.truncated,
            "standalone_question": self.standalone_question,
            "intent": self.intent,
            "notes": self.notes,
            "error": self.error,
        }


def _jsonable(v: Any) -> Any:
    if isinstance(v, Decimal):
        return format(v, "f")
    if isinstance(v, (dt.date, dt.datetime, dt.time)):
        return v.isoformat()
    if isinstance(v, (bytes, bytearray, memoryview)):
        return bytes(v).hex()
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)


def summarize(columns: list[str], rows: list[tuple], row_count: int, truncated: bool) -> str:
    """A deterministic summary built only from the real rows (the model never writes numbers)."""
    if row_count == 0:
        return "No matching records."
    if row_count == 1 and len(columns) == 1:
        return f"{columns[0]}: {sanitize_cell(rows[0][0])}"
    head = f"{row_count} row(s)" + (" (truncated at the row limit)" if truncated else "")
    return head + "\n" + format_rows(columns, rows, 5)


class Engine:
    def __init__(
        self,
        *,
        db: Any,  # promptquery.db.Database (or any object with run_scoped)
        pack: Pack,
        llm: LLMClient,
        tenant_id: str,
        fast_llm: LLMClient | None = None,
        store: ConversationStore | None = None,
        audit: AuditLog | None = None,
        row_cap: int = 1000,
        max_repairs: int = 2,
        timeout_ms: int = 15_000,
        examples: ExampleLibrary | None = None,
        example_k: int = 4,
    ):
        self.db = db
        self.pack = pack
        self.llm = llm
        self.fast_llm = fast_llm or llm
        self.tenant_id = tenant_id
        self.store = store or ConversationStore()
        self.audit = audit or AuditLog(None)
        self.row_cap = row_cap
        self.max_repairs = max_repairs
        self.timeout_ms = timeout_ms
        self.examples = examples
        self.example_k = example_k
        self.system_prompt = build_system_prompt(pack)
        self._allowed = pack.allowed_columns()

    # ------------------------------------------------------------------ sessions
    def new_session(self, session_id: str | None = None) -> Session:
        return self.store.create(self.tenant_id, session_id)

    def reset_session(self, session_id: str) -> None:
        self.store.delete(session_id)

    # ----------------------------------------------------------------------- ask
    def ask(self, session_id: str, text: str) -> TurnResult:
        text = (text or "").strip()
        session = self.store.get(session_id) or self.store.create(self.tenant_id, session_id)
        if session.is_full:
            self.audit.record("turn_limit", session_id=session.id, tenant=session.tenant_id)
            return TurnResult(
                session.id, session.turn_count, "turn_limit",
                answer_text=TURN_LIMIT_MESSAGE, error="TURN_LIMIT",
            )
        turn_no = session.turn_count + 1
        if not text:
            return self._finish(session, turn_no, text, text, "NEW", "error",
                                error="Empty question.")

        started = time.time()
        rewrite, hint = rewrite_followup(self.fast_llm, text, session.turns)
        self.audit.record(
            "rewrite", session_id=session.id, tenant=session.tenant_id, turn_no=turn_no,
            user_text=text, intent=rewrite.intent, standalone=rewrite.standalone_question,
        )
        if rewrite.intent == "CLARIFY":
            return self._finish(session, turn_no, text, rewrite.standalone_question, "CLARIFY",
                                "clarify", answer_text=rewrite.clarification or "")

        base_sql = None
        if rewrite.intent == "REFINE":
            base = next((t for t in session.turns if t.turn_no == rewrite.base_turn), None)
            base = base if base and base.status == "ok" and base.sql else session.last_ok
            base_sql, base_no = (base.sql, base.turn_no) if base else (None, None)
        else:
            base_no = None
        history = None if rewrite.intent == "RESET" else format_history(session.turns)
        shots = (
            render_examples(self.examples.retrieve(rewrite.standalone_question, self.example_k))
            if self.examples else ""
        )
        user_msg = build_user_message(
            question=rewrite.standalone_question, history=history, base_sql=base_sql,
            base_turn=base_no, ordinal_hint=hint, examples=shots,
        )
        result = self._generate_and_run(
            session, turn_no, text, rewrite.standalone_question, rewrite.intent, user_msg
        )
        self.audit.record(
            "turn_done", session_id=session.id, tenant=session.tenant_id, turn_no=turn_no,
            status=result.status, row_count=result.row_count, sql=result.sql,
            error=result.error, notes=result.notes, seconds=round(time.time() - started, 2),
        )
        return result

    # -------------------------------------------------------------- generate + run
    def _generate_and_run(
        self, session: Session, turn_no: int, text: str, standalone: str, intent: str,
        user_msg: str,
    ) -> TurnResult:
        messages: list[dict[str, str]] = [{"role": "user", "content": user_msg}]
        notes: list[str] = []
        last_error = "The model did not produce a valid query."
        empty_candidate: tuple[str, QueryResult] | None = None
        empty_retried = False

        for attempt in range(1 + self.max_repairs):
            try:
                raw = self.llm.generate_messages(self.system_prompt, messages, cache_system=True)
            except Exception as e:  # noqa: BLE001
                last_error = f"Language model unavailable: {e}"
                self.audit.record("llm_error", session_id=session.id, turn_no=turn_no,
                                  attempt=attempt, error=str(e))
                break
            parsed = parse_model_output(raw)
            self.audit.record("generated", session_id=session.id, turn_no=turn_no,
                              attempt=attempt, kind=parsed.kind, plan=parsed.plan,
                              sql=parsed.sql, reason=parsed.reason)

            if parsed.kind == "refuse":
                return self._finish(session, turn_no, text, standalone, intent, "refused",
                                    answer_text=f"I can't help with that: {parsed.reason}",
                                    notes=notes)
            if parsed.kind == "none":
                last_error = "The model reply contained no SQL."
                self._repair(messages, raw, repair_message("format", "", self._as_of()))
                continue

            sql = parsed.sql
            if uses_wall_clock(sql):
                notes.append("rewrote CURRENT_DATE/now() to the as-of date")
            sql = apply_as_of(sql, self.pack.as_of)
            try:
                vq = validate_report_query(
                    sql, allowed_columns=self._allowed, row_cap=self.row_cap,
                    tenant_id=self.tenant_id,
                )
            except UnsafeQuery as e:
                last_error = f"Rejected by the safety validator: {e}"
                self.audit.record("rejected", session_id=session.id, turn_no=turn_no,
                                  attempt=attempt, sql=sql, reason=str(e))
                notes.append(f"validator rejected attempt {attempt + 1}: {e}")
                self._repair(messages, raw, repair_message("unsafe", str(e), self._as_of()))
                continue

            try:
                res = self.db.run_scoped(
                    vq.sql, tenant_id=self.tenant_id, max_rows=self.row_cap,
                    timeout_ms=self.timeout_ms,
                )
            except Exception as e:  # noqa: BLE001 - DB errors feed the repair loop
                last_error = f"Database error: {str(e).strip().splitlines()[0]}"
                self.audit.record("db_error", session_id=session.id, turn_no=turn_no,
                                  attempt=attempt, sql=vq.sql, error=str(e))
                notes.append(f"database error on attempt {attempt + 1}")
                self._repair(messages, raw, repair_message(
                    "db_error", str(e).strip().splitlines()[0], self._as_of()))
                continue

            if not res.rows and not empty_retried and attempt < self.max_repairs:
                # An empty answer is the classic symptom of a wrong status value or period.
                empty_retried = True
                empty_candidate = (vq.sql, res)
                notes.append("empty result: asked the model to double-check filters")
                self._repair(messages, raw, repair_message("empty", "", self._as_of()))
                continue
            return self._ok(session, turn_no, text, standalone, intent, vq.sql, res, notes,
                            parsed.plan)

        if empty_candidate is not None:  # the retry failed; the empty answer stands
            sql, res = empty_candidate
            notes.append("kept the original empty result")
            return self._ok(session, turn_no, text, standalone, intent, sql, res, notes, "")
        return self._finish(session, turn_no, text, standalone, intent, "error",
                            error=last_error, notes=notes)

    # -------------------------------------------------------------------- helpers
    def _as_of(self) -> str:
        return self.pack.as_of.isoformat()

    @staticmethod
    def _repair(messages: list[dict[str, str]], raw: str, feedback: str) -> None:
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": feedback})

    def _ok(
        self, session: Session, turn_no: int, text: str, standalone: str, intent: str,
        sql: str, res: QueryResult, notes: list[str], plan: str,
    ) -> TurnResult:
        if res.truncated:
            notes.append(f"result truncated to the first {self.row_cap} rows")
        if plan:
            notes.insert(0, f"plan: {plan}")
        return self._finish(
            session, turn_no, text, standalone, intent, "ok", sql=sql, columns=res.columns,
            rows=res.rows, truncated=res.truncated, notes=notes,
            answer_text=summarize(res.columns, res.rows, len(res.rows), res.truncated),
        )

    def _finish(
        self, session: Session, turn_no: int, text: str, standalone: str, intent: str,
        status: str, *, sql: str | None = None, columns: list[str] | None = None,
        rows: list[tuple] | None = None, truncated: bool = False, answer_text: str = "",
        error: str | None = None, notes: list[str] | None = None,
    ) -> TurnResult:
        columns = columns or []
        rows = rows or []
        session.turns.append(TurnRecord(
            turn_no=turn_no, user_text=text, standalone_question=standalone, intent=intent,
            status=status, sql=sql, columns=columns, preview_rows=[tuple(r) for r in rows[:PREVIEW_ROWS]],
            row_count=len(rows), truncated=truncated, answer_text=answer_text, error=error,
        ))
        return TurnResult(
            session.id, turn_no, status, answer_text=answer_text if status != "error" else (error or ""),
            sql=sql, columns=columns, rows=rows, row_count=len(rows), truncated=truncated,
            standalone_question=standalone, intent=intent, notes=notes or [], error=error,
        )
