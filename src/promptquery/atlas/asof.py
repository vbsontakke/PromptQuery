"""Make "today" deterministic.

Atlas's demo data ends on a fixed seed date, while the machine clock moves on. A question like
"last month" must be answered relative to a configured *as-of* date, not the wall clock, or the
same question returns different (usually empty) results every day. The prompt already tells the
model never to call ``CURRENT_DATE``/``now()``; this is the safety net for when it does anyway.
"""

from __future__ import annotations

import datetime as dt

import sqlglot
from sqlglot import exp


def _date_literal(as_of: dt.date) -> exp.Expression:
    return exp.Cast(this=exp.Literal.string(as_of.isoformat()), to=exp.DataType.build("date"))


def _timestamp_literal(as_of: dt.date) -> exp.Expression:
    return exp.Cast(
        this=exp.Literal.string(f"{as_of.isoformat()} 00:00:00"),
        to=exp.DataType.build("timestamp"),
    )


def apply_as_of(sql: str, as_of: dt.date) -> str:
    """Replace CURRENT_DATE / now() / CURRENT_TIMESTAMP / LOCALTIMESTAMP with the as-of date.

    Returns the SQL unchanged if it cannot be parsed (the validator reports that error).
    """
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception:
        return sql
    if tree is None:
        return sql

    def swap(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.CurrentDate):
            return _date_literal(as_of)
        if isinstance(node, (exp.CurrentTimestamp, exp.Localtimestamp)):
            return _timestamp_literal(as_of)
        return node

    return tree.transform(swap).sql(dialect="postgres")


def uses_wall_clock(sql: str) -> bool:
    """True if the SQL references the wall clock (used to flag/log model slips)."""
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception:
        return False
    return tree is not None and any(
        isinstance(n, (exp.CurrentDate, exp.CurrentTimestamp, exp.Localtimestamp))
        for n in tree.walk()
    )
