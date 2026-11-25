"""Verify the curated pack against a live, seeded Atlas database (skipped without one)."""

import datetime as dt
import os

import pytest

from promptquery.atlas.pack_loader import load_pack
from promptquery.db import Database
from promptquery.safety import validate_report_query
from promptquery.schema import introspect

OWNER = os.environ.get("PQ_TEST_OWNER_DSN")
BOT = os.environ.get("PQ_TEST_BOT_DSN")
AS_OF = dt.date(2026, 7, 6)

pytestmark = pytest.mark.skipif(
    not (OWNER and BOT), reason="PQ_TEST_OWNER_DSN / PQ_TEST_BOT_DSN not set"
)


@pytest.fixture(scope="module")
def owner():
    with Database(OWNER) as db:
        yield db


@pytest.fixture(scope="module")
def pack(owner):
    return load_pack(introspect(owner), AS_OF, strict=True)  # strict: raises on any drift


def test_pack_matches_database_exactly(pack):
    assert len(pack.tables) >= 100


def test_every_enum_value_in_data_is_declared_in_the_pack(owner, pack):
    """The pack must never hide a real value, or the model cannot filter on it."""
    missing = []
    for t in pack.tables.values():
        for c in t.columns:
            if not c.enum_values:
                continue
            _, rows = owner.execute(
                f'SELECT DISTINCT "{c.name}" FROM "{t.name}" WHERE "{c.name}" IS NOT NULL'
            )
            for (v,) in rows:
                if v not in c.enum_values:
                    missing.append(f"{t.name}.{c.name}={v!r}")
    assert not missing, missing


def test_inferred_joins_have_no_orphans(owner, pack):
    """Every inferred *_id -> table.id join must resolve for all non-null values."""
    bad = []
    for t in pack.tables.values():
        for c in t.columns:
            if not (c.ref_inferred and c.ref and "[" not in c.ref and "|" not in c.ref):
                continue
            target, tcol = c.ref.split(".")
            _, rows = owner.execute(
                f'SELECT count(*) FROM "{t.name}" x WHERE x."{c.name}" IS NOT NULL AND NOT EXISTS '
                f'(SELECT 1 FROM "{target}" y WHERE y."{tcol}" = x."{c.name}")'
            )
            if rows[0][0]:
                bad.append(f"{t.name}.{c.name} -> {c.ref}: {rows[0][0]} orphans")
    assert not bad, bad


def test_polymorphic_partner_join_resolves(owner):
    _, rows = owner.execute(
        "SELECT count(*) FROM fin_journal_lines l WHERE l.partner_id IS NOT NULL AND NOT ("
        "(l.partner_type = 'CUSTOMER' AND EXISTS (SELECT 1 FROM sales_customers c WHERE c.id = l.partner_id)) OR "
        "(l.partner_type = 'VENDOR' AND EXISTS (SELECT 1 FROM proc_vendors v WHERE v.id = l.partner_id)))"
    )
    assert rows[0][0] == 0


def test_every_metric_sql_validates_and_runs_for_every_tenant(owner, pack):
    _, tenants = owner.execute("SELECT slug, id FROM adm_tenants ORDER BY slug")
    with Database(BOT) as bot:
        for key, m in pack.metrics.items():
            sql = m.sql.replace("{as_of}", AS_OF.isoformat())
            vq = validate_report_query(sql, allowed_columns=pack.allowed_columns())
            for slug, tid in tenants:
                try:
                    bot.run_scoped(vq.sql, tenant_id=tid)
                except Exception as e:  # pragma: no cover - failure path
                    pytest.fail(f"metric {key!r} failed for tenant {slug}: {e}")


def test_pack_renders_within_prompt_budget(pack):
    text = pack.render_schema() + pack.render_metrics() + pack.render_joins()
    assert len(text) // 4 < 20_000  # whole curated pack stays well under 20k tokens
