import pytest

from promptquery.safety import UnsafeQuery, validate_report_query, validate_select_only

ALLOWED = {
    "sales_orders": ["id", "tenant_id", "order_number", "status", "total_amount", "order_date",
                     "customer_id"],
    "sales_customers": ["id", "tenant_id", "name", "customer_code"],
    "hr_employees": ["id", "tenant_id", "first_name", "last_name"],  # no salary / PII columns
}


def v(sql, **kw):
    return validate_report_query(sql, allowed_columns=ALLOWED, **kw)


# ---- accepted ---------------------------------------------------------------------------

@pytest.mark.parametrize("sql", [
    "SELECT status, COUNT(*) FROM sales_orders GROUP BY status",
    "SELECT c.name, SUM(o.total_amount) FROM sales_orders o "
    "JOIN sales_customers c ON c.id = o.customer_id GROUP BY c.name ORDER BY 2 DESC",
    "WITH t AS (SELECT customer_id, SUM(total_amount) a FROM sales_orders GROUP BY 1) "
    "SELECT * FROM t WHERE a > 0",
    "SELECT DATE_TRUNC('month', order_date) m, COUNT(*) FROM sales_orders GROUP BY 1",
    "SELECT * FROM public.sales_orders",
    "SELECT first_name FROM hr_employees",
    "SELECT g FROM generate_series(1, 3) g",
    "SELECT status FROM sales_orders UNION SELECT 'x'",
    "SELECT age(order_date) FROM sales_orders",
])
def test_accepts_legitimate_reporting_sql(sql):
    v(sql)


def test_cte_name_shadowing_is_not_mistaken_for_a_table():
    r = v("WITH recent AS (SELECT id FROM sales_orders) SELECT * FROM recent")
    assert r.tables == {"sales_orders"}


# ---- rejected: tables / schemas ---------------------------------------------------------

@pytest.mark.parametrize("sql", [
    "SELECT * FROM core_users",
    "SELECT * FROM hr_payroll_run_lines",
    "SELECT * FROM pg_catalog.pg_tables",
    "SELECT * FROM information_schema.tables",
    "SELECT * FROM other.sales_orders",
    "SELECT * FROM db.public.sales_orders",
    "SELECT * FROM sales_orders o JOIN core_users u ON u.id = o.customer_id",
    "SELECT (SELECT count(*) FROM core_users)",
    "WITH x AS (SELECT * FROM core_users) SELECT * FROM x",
])
def test_rejects_tables_outside_allowlist(sql):
    with pytest.raises(UnsafeQuery):
        v(sql)


# ---- rejected: columns ------------------------------------------------------------------

@pytest.mark.parametrize("sql", [
    "SELECT base_salary FROM hr_employees",
    "SELECT e.national_id FROM hr_employees e",
    "SELECT nosuchcolumn FROM sales_orders",
])
def test_rejects_denied_or_unknown_columns(sql):
    with pytest.raises(UnsafeQuery):
        v(sql)


def test_star_on_pii_table_expands_to_readable_columns_only():
    r = v("SELECT * FROM hr_employees")
    assert "base_salary" not in r.sql and "national_id" not in r.sql
    assert "first_name" in r.sql


def test_qualified_star_expands_too():
    r = v("SELECT e.* FROM hr_employees e")
    assert "salary" not in r.sql and "last_name" in r.sql


def test_unnamed_outputs_keep_their_names_when_there_is_no_star():
    r = v("SELECT COUNT(*) FROM sales_orders")
    assert "_col_" not in r.sql


# ---- rejected: statements / functions ---------------------------------------------------

@pytest.mark.parametrize("sql", [
    "SELECT * INTO newt FROM sales_orders",
    "SELECT * FROM sales_orders FOR UPDATE",
    "SELECT * FROM sales_orders FOR SHARE",
    "SET statement_timeout = 0",
    "BEGIN READ WRITE",
    "COMMIT",
    "COPY sales_orders TO '/tmp/x'",
    "SHOW search_path",
    "EXPLAIN ANALYZE DELETE FROM sales_orders",
    "DELETE FROM sales_orders",
    "SELECT 1; SELECT 2",
    "WITH d AS (DELETE FROM sales_orders RETURNING id) SELECT * FROM d",
])
def test_rejects_non_select_constructs(sql):
    with pytest.raises(UnsafeQuery):
        v(sql)


@pytest.mark.parametrize("sql", [
    "SELECT pg_sleep(5)",
    "SELECT pg_read_file('/etc/passwd')",
    "SELECT pg_catalog.pg_read_file('/etc/passwd')",
    "SELECT pg_ls_dir('/')",
    "SELECT query_to_xml('select * from core_users', true, true, '')",
    "SELECT nextval('s')",
    "SELECT current_setting('app.tenant_id')",
    "SELECT set_config('app.tenant_id', 'x', false)",
    "SELECT lo_import('/etc/passwd')",
    "SELECT dblink('host=x', 'select 1')",
    "SELECT txid_current()",
    "SELECT inet_server_addr()",
    "SELECT * FROM pg_ls_dir('/')",
])
def test_rejects_dangerous_or_unknown_functions(sql):
    with pytest.raises(UnsafeQuery):
        v(sql)


def test_tenant_setting_cannot_be_changed_or_read_via_sql():
    for sql in ("SELECT set_config('app.tenant_id', 'x', false)", "SELECT current_setting('app.tenant_id')"):
        with pytest.raises(UnsafeQuery):
            v(sql)


def test_rejects_oversized_sql():
    with pytest.raises(UnsafeQuery):
        v("SELECT " + ", ".join(["1"] * 20000))


# ---- LIMIT enforcement and normalisation -------------------------------------------------

def test_injects_limit_when_missing():
    r = v("SELECT id FROM sales_orders", row_cap=500)
    assert r.limit_applied and r.sql.rstrip().upper().endswith("LIMIT 500")


def test_keeps_smaller_limit_untouched():
    r = v("SELECT id FROM sales_orders LIMIT 10", row_cap=500)
    assert not r.limit_applied and r.sql.upper().endswith("LIMIT 10")


def test_clamps_larger_limit():
    r = v("SELECT id FROM sales_orders LIMIT 99999", row_cap=500)
    assert r.limit_applied and r.sql.upper().endswith("LIMIT 500")


def test_limit_all_is_clamped():
    r = v("SELECT id FROM sales_orders LIMIT ALL", row_cap=500)
    assert r.sql.upper().endswith("LIMIT 500")


def test_limit_applies_to_whole_union():
    r = v("SELECT id FROM sales_orders UNION ALL SELECT id FROM sales_customers", row_cap=7)
    assert r.sql.upper().endswith("LIMIT 7")


def test_comments_are_stripped_from_executed_sql():
    r = v("SELECT id FROM sales_orders /* ; DROP TABLE x */ -- trailing; DROP")
    assert "DROP" not in r.sql.upper() and "--" not in r.sql and "/*" not in r.sql


# ---- hardened base validator -------------------------------------------------------------

@pytest.mark.parametrize("sql", [
    "SELECT * INTO t FROM users",
    "SELECT * FROM users FOR UPDATE",
    "SELECT pg_sleep(1)",
    "SELECT pg_read_file('x')",
    "SELECT nextval('s')",
])
def test_base_validator_now_rejects_into_lock_and_more_functions(sql):
    with pytest.raises(UnsafeQuery):
        validate_select_only(sql)


# ---- tenant scoping (SQL-level, independent of Postgres RLS) -----------------------------

TENANT = "ab16b03f-aca2-407c-9f5a-839bd09b4e52"


def scoped(sql):
    return validate_report_query(sql, allowed_columns=ALLOWED, tenant_id=TENANT).sql


def _filters(sql):
    return sql.count(f"tenant_id = '{TENANT}'")


def test_every_table_reference_is_wrapped_in_a_tenant_filter():
    assert _filters(scoped("SELECT COUNT(*) FROM sales_orders")) == 1
    j = scoped("SELECT c.name FROM sales_orders o JOIN sales_customers c ON c.id = o.customer_id")
    assert _filters(j) == 2
    sub = scoped("SELECT id FROM sales_orders WHERE total_amount > (SELECT AVG(total_amount) FROM sales_orders)")
    assert _filters(sub) == 2
    u = scoped("SELECT id FROM sales_orders UNION ALL SELECT id FROM sales_customers")
    assert _filters(u) == 2
    cte = scoped("WITH t AS (SELECT * FROM sales_orders) SELECT COUNT(*) FROM t")
    assert _filters(cte) == 1  # the CTE body is scoped; `t` itself is not a base table


def test_self_join_scopes_both_references_and_keeps_aliases():
    s = scoped("SELECT a.id FROM hr_employees a JOIN hr_employees b ON a.id = b.id")
    assert _filters(s) == 2 and "AS a" in s and "AS b" in s


def test_scoped_subquery_lists_only_readable_columns():
    s = scoped("SELECT first_name FROM hr_employees")
    assert "base_salary" not in s and "national_id" not in s and "*" not in s.split("WHERE")[0]


def test_tenant_id_must_be_a_uuid():
    with pytest.raises(ValueError):
        validate_report_query("SELECT id FROM sales_orders", allowed_columns=ALLOWED, tenant_id="x' OR 1=1 --")


def test_a_hostile_tenant_predicate_in_the_sql_cannot_widen_the_scope():
    s = scoped("SELECT id FROM sales_orders WHERE tenant_id = 'ffffffff-ffff-ffff-ffff-ffffffffffff' OR 1 = 1")
    # our filter sits INSIDE the derived table, so the user's OR cannot remove it
    assert s.index(f"tenant_id = '{TENANT}'") < s.index("OR 1 = 1")


def test_unscoped_when_no_tenant_given():
    assert "tenant_id =" not in validate_report_query(
        "SELECT id FROM sales_orders", allowed_columns=ALLOWED
    ).sql


# ---- FETCH FIRST is a limit too (regression: it used to be replaced by LIMIT <cap>) -------

def test_fetch_first_within_the_cap_is_preserved_exactly():
    r = v("SELECT id FROM sales_orders ORDER BY total_amount DESC FETCH FIRST 3 ROWS ONLY", row_cap=500)
    assert "FETCH FIRST 3 ROWS ONLY" in r.sql.upper() and "LIMIT" not in r.sql.upper()
    assert not r.limit_applied


def test_fetch_next_row_only_is_preserved():
    r = v("SELECT id FROM sales_orders ORDER BY id FETCH NEXT ROW ONLY", row_cap=500)
    assert "FETCH NEXT 1 ROWS ONLY" in r.sql.upper() or "FETCH NEXT ROW" in r.sql.upper()
    assert "LIMIT" not in r.sql.upper()


def test_fetch_first_larger_than_the_cap_is_clamped():
    r = v("SELECT id FROM sales_orders FETCH FIRST 99999 ROWS ONLY", row_cap=500)
    assert r.sql.upper().endswith("LIMIT 500") and r.limit_applied


def test_fetch_with_ties_and_percent_are_left_alone_not_mangled():
    for tail, needle in [("FETCH FIRST 3 ROWS WITH TIES", "WITH TIES"), ("FETCH FIRST 10 PERCENT ROWS ONLY", "PERCENT")]:
        r = v(f"SELECT id FROM sales_orders ORDER BY total_amount {tail}", row_cap=500)
        assert needle in r.sql.upper() and "LIMIT" not in r.sql.upper()
