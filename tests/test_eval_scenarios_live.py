"""Quality gate for the scenario set itself: golds must be valid, safe, non-empty, unambiguous."""

import os

import pytest

from eval.atlas.scenario import load_scenarios
from promptquery.bootstrap import Settings, build_engine
from tests.test_engine import Scripted

OWNER = os.environ.get("PQ_TEST_OWNER_DSN")
BOT = os.environ.get("PQ_TEST_BOT_DSN")


def test_scenario_files_parse_and_are_well_formed():
    scs = load_scenarios()
    assert scs
    ids = [s.id for s in scs]
    assert len(ids) == len(set(ids))
    for s in scs:
        assert 1 <= len(s.turns) <= 6 and s.split in ("dev", "heldout")
        assert s.modules, f"{s.id}: needs a module tag"


def test_every_distinct_question_has_one_gold_answer():
    """The oracle (and the cache) key on the question text, so the text must identify the gold."""
    seen: dict[str, str] = {}
    for s in load_scenarios():
        for t in s.turns:
            key = (t.gold_sql or "REFUSE").strip()
            if t.user in seen and seen[t.user] != key:
                pytest.fail(f"question {t.user!r} has two different golds")
            seen[t.user] = key


@pytest.mark.skipif(not (OWNER and BOT), reason="needs a seeded Atlas DB")
def test_every_gold_validates_runs_and_is_non_empty():
    engines: dict[str, object] = {}
    problems = []
    for s in load_scenarios():
        eng = engines.get(s.tenant) or build_engine(
            Settings(dsn=BOT, tenant=s.tenant, admin_dsn=OWNER), llm=Scripted([]), fast_llm=Scripted([])
        )
        engines[s.tenant] = eng
        from promptquery.safety import validate_report_query

        for i, t in enumerate(s.turns, 1):
            if t.expect != "ok":
                continue
            try:
                vq = validate_report_query(
                    t.gold_sql, allowed_columns=eng.pack.allowed_columns(), row_cap=100_000,
                    tenant_id=eng.tenant_id,
                )
                res = eng.db.run_scoped(vq.sql, tenant_id=eng.tenant_id, max_rows=100_000)
            except Exception as e:  # noqa: BLE001
                problems.append(f"{s.id}#{i}: {str(e).splitlines()[0]}")
                continue
            if not res.rows and not t.allow_empty:
                problems.append(f"{s.id}#{i}: gold returns no rows (an empty gold accepts any empty answer)")
    assert not problems, "\n".join(problems)
