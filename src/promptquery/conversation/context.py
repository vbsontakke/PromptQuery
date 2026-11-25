"""Render conversation history for the model, treating database content as untrusted data.

Result rows come from the database, and database text (a customer name, a note field) can
contain instructions. Rows are therefore fenced in <prior_result untrusted="true"> blocks,
stripped of control characters and backticks, and length-capped. The system prompt states
that anything inside such a block is data, never an instruction.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Sequence
from decimal import Decimal

from .state import TurnRecord

_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
CELL_MAX_CHARS = 80
FULL_TURNS = 3  # older turns are shown as a single line


def sanitize_cell(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, Decimal):
        value = value.normalize() if value == value.to_integral() else value
        text = format(value, "f")
    elif isinstance(value, (dt.date, dt.datetime)):
        text = value.isoformat()
    else:
        text = str(value)
    text = _CTRL.sub(" ", text).replace("`", "'").replace("<", "(").replace(">", ")")
    text = " ".join(text.split())
    return text if len(text) <= CELL_MAX_CHARS else text[: CELL_MAX_CHARS - 1] + "…"


def format_rows(columns: Sequence[str], rows: Sequence[Sequence[object]], limit: int) -> str:
    lines = [" | ".join(columns)]
    for i, row in enumerate(rows[:limit], start=1):
        lines.append(f"{i}. " + " | ".join(sanitize_cell(v) for v in row))
    return "\n".join(lines)


def format_turn(t: TurnRecord, *, full: bool, row_limit: int = 20) -> str:
    head = f"[Turn {t.turn_no}] user asked: {t.user_text!r}"
    if t.standalone_question and t.standalone_question != t.user_text:
        head += f"\n  understood as: {t.standalone_question}"
    if t.status != "ok":
        detail = t.error or t.answer_text or ""
        return f"{head}\n  outcome: {t.status}{(' - ' + detail) if detail else ''}"
    if not full:
        return f"{head}\n  returned {t.row_count} row(s); columns: {', '.join(t.columns)}"
    body = format_rows(t.columns, t.preview_rows, row_limit)
    more = "" if t.row_count <= len(t.preview_rows) else f"\n(first {len(t.preview_rows)} of {t.row_count} rows shown)"
    return (
        f"{head}\n  SQL: {t.sql}\n"
        f'  <prior_result untrusted="true">\n{body}{more}\n  </prior_result>'
    )


def format_history(turns: Sequence[TurnRecord], *, full_turns: int = FULL_TURNS) -> str:
    if not turns:
        return "(no earlier turns)"
    cutoff = len(turns) - full_turns
    return "\n\n".join(format_turn(t, full=i >= cutoff) for i, t in enumerate(turns))


_ORDINALS = {
    "first": 1, "1st": 1, "top": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4, "fifth": 5, "5th": 5, "last": -1, "bottom": -1,
}
_ORDINAL_RE = re.compile(
    r"\b(?:the\s+)?(first|1st|top|second|2nd|third|3rd|fourth|4th|fifth|5th|last|bottom)"
    r"(?:\s+(?:one|row|result|entry|record|item|customer|vendor|account|invoice|order|employee|"
    r"product|department|project|warehouse))?\b",
    re.IGNORECASE,
)


def resolve_ordinal(message: str, prev: TurnRecord | None) -> str | None:
    """Deterministically resolve 'the first one' / 'the last customer' against the previous result.

    Returns a one-line hint naming the referenced row, or None when there is no ordinal
    reference or no previous result to resolve it against.
    """
    if prev is None or not prev.preview_rows:
        return None
    m = _ORDINAL_RE.search(message)
    if not m:
        return None
    n = _ORDINALS[m.group(1).lower()]
    rows = prev.preview_rows
    if n == -1:
        if prev.row_count > len(rows):  # the true last row is beyond the preview: do not guess
            return None
        idx = len(rows) - 1
    else:
        idx = n - 1
    if idx >= len(rows):
        return None
    cells = ", ".join(f"{c}={sanitize_cell(v)}" for c, v in zip(prev.columns, rows[idx], strict=False))
    return f"'{m.group(0)}' refers to row {idx + 1} of turn {prev.turn_no}'s result: {cells}"
