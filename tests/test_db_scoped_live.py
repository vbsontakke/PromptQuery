"""Live checks of the bot database boundary. Skipped unless an Atlas DB is configured.

    PQ_TEST_OWNER_DSN=postgresql://atlas@127.0.0.1:55440/atlas
    PQ_TEST_BOT_DSN=postgresql://atlas_bot:botpw@127.0.0.1:55440/atlas
"""

import os

import psycopg
import pytest

from promptquery.db import Database

OWNER = os.environ.get("PQ_TEST_OWNER_DSN")
BOT = os.environ.get("PQ_TEST_BOT_DSN")

pytestmark = pytest.mark.skipif(
    not (OWNER and BOT), reason="PQ_TEST_OWNER_DSN / PQ_TEST_BOT_DSN not set"
)


@pytest.fixture(scope="module")
def tenants():
    with Database(OWNER) as db:
        _, rows = db.execute("SELECT slug, id FROM adm_tenants ORDER BY slug")
    return {slug: str(i) for slug, i in rows}


@pytest.fixture()
def bot():
    with Database(BOT) as db:
        yield db


def test_tenants_are_isolated(bot, tenants):
    a = bot.run_scoped("SELECT count(*) FROM sales_orders", tenant_id=tenants["acme"])
    m = bot.run_scoped("SELECT count(*) FROM sales_orders", tenant_id=tenants["manufacturing"])
    assert a.rows[0][0] > 0 and m.rows[0][0] > 0
    with Database(OWNER) as owner:
        _, rows = owner.execute("SELECT count(*) FROM sales_orders")
    assert a.rows[0][0] + m.rows[0][0] < rows[0][0]  # the owner sees every tenant


def test_explicit_cross_tenant_filter_returns_nothing(bot, tenants):
    r = bot.run_scoped(
        f"SELECT count(*) FROM sales_orders WHERE tenant_id = '{tenants['manufacturing']}'",
        tenant_id=tenants["acme"],
    )
    assert r.rows[0][0] == 0


def test_missing_tenant_fails_closed(bot):
    with pytest.raises(psycopg.Error):
        bot.run_scoped("SELECT count(*) FROM sales_orders", tenant_id=None)


def test_denied_table_and_columns(bot, tenants):
    t = tenants["acme"]
    for sql in (
        "SELECT * FROM core_users",
        "SELECT * FROM hr_payroll_run_lines",
        "SELECT base_salary FROM hr_employees",
        "SELECT * FROM hr_employees",
    ):
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            bot.run_scoped(sql, tenant_id=t)


def test_allowed_pii_table_columns_work(bot, tenants):
    r = bot.run_scoped("SELECT first_name, last_name FROM hr_employees", tenant_id=tenants["acme"])
    assert r.rows


def test_row_cap_truncates(bot, tenants):
    r = bot.run_scoped(
        "SELECT * FROM fin_journal_lines", tenant_id=tenants["acme"], max_rows=5
    )
    assert len(r.rows) == 5 and r.truncated


def test_row_cap_not_truncated_when_exact(bot, tenants):
    total = bot.run_scoped("SELECT count(*) FROM fin_accounts", tenant_id=tenants["acme"]).rows[0][0]
    r = bot.run_scoped("SELECT * FROM fin_accounts", tenant_id=tenants["acme"], max_rows=total)
    assert len(r.rows) == total and not r.truncated


def test_writes_are_blocked_even_if_validator_is_bypassed(bot, tenants):
    for sql in ("DELETE FROM sales_orders", "UPDATE fin_accounts SET name = 'x'"):
        with pytest.raises(psycopg.Error):
            bot.run_scoped(sql, tenant_id=tenants["acme"])


def test_timeout_is_enforced(bot, tenants):
    with pytest.raises(psycopg.errors.QueryCanceled):
        bot.run_scoped("SELECT pg_sleep(5)", tenant_id=tenants["acme"], timeout_ms=200)


def test_connection_usable_after_error_and_tenant_does_not_leak(bot, tenants):
    with pytest.raises(psycopg.Error):
        bot.run_scoped("SELECT * FROM core_users", tenant_id=tenants["acme"])
    # a later call with no tenant must not inherit the previous call's setting
    with pytest.raises(psycopg.Error):
        bot.run_scoped("SELECT count(*) FROM sales_orders", tenant_id=None)
    ok = bot.run_scoped("SELECT count(*) FROM sales_orders", tenant_id=tenants["acme"])
    assert ok.rows[0][0] > 0


def test_invalid_tenant_id_rejected_before_reaching_sql(bot):
    with pytest.raises(ValueError):
        bot.run_scoped("SELECT 1", tenant_id="x'; drop table sales_orders; --")


def test_sql_level_scoping_isolates_tenants_even_without_rls(tenants):
    """Connected as the table OWNER (RLS does not apply), the injected scope alone isolates."""
    from promptquery.atlas.security import allowed_columns, allowed_tables
    from promptquery.safety import validate_report_query
    from promptquery.schema import introspect

    with Database(OWNER) as owner:
        schema = introspect(owner)
        allowed = {t: allowed_columns(schema, t) for t in allowed_tables(schema)}
        everyone = owner.run_scoped("SELECT count(*) FROM sales_orders", tenant_id=None).rows[0][0]
        for slug in ("acme", "manufacturing"):
            tid = tenants[slug]
            q = validate_report_query(
                "SELECT count(*) FROM sales_orders", allowed_columns=allowed, tenant_id=tid
            )
            mine = owner.run_scoped(q.sql, tenant_id=None).rows[0][0]
            assert 0 < mine < everyone
        # a join across scoped tables never leaks another tenant's rows
        q = validate_report_query(
            "SELECT count(*) FROM sales_orders o JOIN sales_customers c ON c.id = o.customer_id "
            f"WHERE o.tenant_id <> '{tenants['acme']}'",
            allowed_columns=allowed, tenant_id=tenants["acme"],
        )
        assert owner.run_scoped(q.sql, tenant_id=None).rows[0][0] == 0
