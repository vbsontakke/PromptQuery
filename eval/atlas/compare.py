"""Execution-accuracy comparison of a candidate result set against the gold result set.

A candidate is correct when it contains the same *information* as gold:

* every gold column is present (matched by its values, not its name, so ``SUM(x) AS total`` and
  ``SUM(x) AS revenue`` are the same answer; column order is irrelevant);
* the rows agree as a multiset (or as a sequence when ``order=True``, e.g. top-N rankings);
* numbers agree within a small tolerance, dates/UUIDs/strings are compared canonically;
* extra candidate columns (typically an identifying name next to a measure) are allowed unless
  ``extra_columns=False``.
"""

from __future__ import annotations

import datetime as dt
import math
import uuid
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

REL_TOL = 1e-4
ABS_TOL = 0.005


@dataclass(frozen=True)
class CompareOptions:
    order: bool = False  # rows must appear in the same sequence as gold
    extra_columns: bool = True  # tolerate columns beyond gold's
    rel_tol: float = REL_TOL
    abs_tol: float = ABS_TOL


@dataclass(frozen=True)
class CompareResult:
    ok: bool
    reason: str = ""


def norm(v: Any) -> Any:
    """Canonical form: numbers -> float, temporal -> ISO text, UUID/other -> stripped text."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float, Decimal)):
        return float(v)
    if isinstance(v, dt.datetime):
        return v.isoformat()
    if isinstance(v, dt.date):
        return v.isoformat()
    if isinstance(v, uuid.UUID):
        return str(v)
    return str(v).strip()


def _key(v: Any) -> tuple:
    """Sortable, tolerance-friendly key (numbers bucketed to cents)."""
    if v is None:
        return (0, 0)
    if isinstance(v, bool):
        return (1, int(v))
    if isinstance(v, float):
        return (2, round(v, 2))
    return (3, v)


def _cell_eq(a: Any, b: Any, o: CompareOptions) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=o.rel_tol, abs_tol=o.abs_tol)
    return a == b


def _col_sig(col: list[Any]) -> list[tuple]:
    return sorted(_key(v) for v in col)


def _match_columns(
    gold_cols: list[list[Any]], cand_cols: list[list[Any]], o: CompareOptions
) -> list[int] | None:
    """For each gold column choose a distinct candidate column holding the same values."""
    used: set[int] = set()
    chosen: list[int] = []
    for g in gold_cols:
        gs = _col_sig(g)
        hit = None
        for k, c in enumerate(cand_cols):
            if k in used:
                continue
            cs = _col_sig(c)
            if len(cs) == len(gs) and all(
                _cell_eq(x[1], y[1], o) if x[0] == y[0] == 2 else x == y for x, y in zip(gs, cs, strict=True)
            ):
                hit = k
                break
        if hit is None:
            return None
        used.add(hit)
        chosen.append(hit)
    return chosen


def compare_results(
    gold_rows: list[tuple], cand_rows: list[tuple], options: CompareOptions | None = None
) -> CompareResult:
    o = options or CompareOptions()
    g = [tuple(norm(v) for v in r) for r in gold_rows]
    c = [tuple(norm(v) for v in r) for r in cand_rows]

    if len(g) != len(c):
        return CompareResult(False, f"row count differs: expected {len(g)}, got {len(c)}")
    if not g:
        return CompareResult(True, "both empty")

    gw, cw = len(g[0]), len(c[0])
    if cw < gw:
        return CompareResult(False, f"too few columns: expected {gw}, got {cw}")
    if cw > gw and not o.extra_columns:
        return CompareResult(False, f"extra columns: expected {gw}, got {cw}")

    gold_cols = [[r[j] for r in g] for j in range(gw)]
    cand_cols = [[r[j] for r in c] for j in range(cw)]
    mapping = _match_columns(gold_cols, cand_cols, o)
    if mapping is None:
        return CompareResult(False, "no candidate column set reproduces the gold values")

    projected = [tuple(r[k] for k in mapping) for r in c]
    if o.order:
        for i, (a, b) in enumerate(zip(g, projected, strict=True)):
            if not all(_cell_eq(x, y, o) for x, y in zip(a, b, strict=True)):
                return CompareResult(False, f"row {i + 1} differs: expected {a}, got {b}")
        return CompareResult(True)

    ga = Counter(tuple(_key(v) for v in r) for r in g)
    ca = Counter(tuple(_key(v) for v in r) for r in projected)
    if ga == ca:
        return CompareResult(True)
    # tolerance fallback: pair rows after sorting by key
    gs = sorted(g, key=lambda r: tuple(_key(v) for v in r))
    cs = sorted(projected, key=lambda r: tuple(_key(v) for v in r))
    for a, b in zip(gs, cs, strict=True):
        if not all(_cell_eq(x, y, o) for x, y in zip(a, b, strict=True)):
            return CompareResult(False, f"rows differ: expected {a}, got {b}")
    return CompareResult(True)
