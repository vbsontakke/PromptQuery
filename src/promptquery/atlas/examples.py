"""Verified question -> SQL examples, retrieved by similarity and shown as patterns.

A handful of close, verified examples is the cheapest accuracy lever after the schema pack: it
shows the model the house style (status filters, `posting_date` ranges, tiebreakers, how a
refusal looks) on questions shaped like the one being asked.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..retrieval import tokenize

DEFAULT_PATH = Path(__file__).parent / "pack" / "examples.yaml"
MIN_SCORE = 0.12  # below this an example is more distraction than help


@dataclass(frozen=True)
class Example:
    question: str
    sql: str | None = None
    refuse: str | None = None
    tenant: str = "acme"


def _vec(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    tf = Counter(tokens)
    return {t: c * idf.get(t, 0.0) for t, c in tf.items()}


def _norm(v: dict[str, float]) -> float:
    return math.sqrt(sum(x * x for x in v.values()))


class ExampleLibrary:
    def __init__(self, examples: list[Example]):
        self.examples = examples
        docs = [tokenize(e.question) for e in examples]
        n = max(len(docs), 1)
        df: Counter[str] = Counter(t for d in docs for t in set(d))
        self._idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
        self._vecs = [_vec(d, self._idf) for d in docs]

    @classmethod
    def load(cls, path: Path | None = None) -> "ExampleLibrary":
        raw = yaml.safe_load((path or DEFAULT_PATH).read_text(encoding="utf-8")) or {}
        out = []
        for e in raw.get("examples", []):
            out.append(Example(
                question=e["q"].strip(),
                sql=e["sql"].strip() if e.get("sql") else None,
                refuse=e.get("refuse"),
                tenant=e.get("tenant", "acme"),
            ))
        return cls(out)

    def retrieve(self, question: str, k: int = 4) -> list[Example]:
        q = _vec(tokenize(question), self._idf)
        qn = _norm(q)
        if qn == 0:
            return []
        scored = []
        for ex, v in zip(self.examples, self._vecs, strict=True):
            vn = _norm(v)
            if vn == 0:
                continue
            dot = sum(w * v.get(t, 0.0) for t, w in q.items())
            scored.append((dot / (qn * vn), ex))
        scored.sort(key=lambda x: -x[0])
        return [ex for s, ex in scored[:k] if s >= MIN_SCORE]


def render_examples(examples: list[Example]) -> str:
    """Prompt text for retrieved examples ('' when there are none)."""
    if not examples:
        return ""
    blocks = []
    for e in examples:
        if e.sql:
            blocks.append(f"Q: {e.question}\n```sql\n{e.sql}\n```")
        else:
            blocks.append(f"Q: {e.question}\nREFUSE: {e.refuse}")
    return (
        "Verified examples of similar questions (patterns for style, status filters and dates; "
        "adapt the entities, periods and columns to the question, do not copy values):\n\n"
        + "\n\n".join(blocks)
    )
