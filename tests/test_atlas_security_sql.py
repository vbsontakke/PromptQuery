from promptquery.atlas.security import (
    DENY_TABLES,
    allowed_columns,
    allowed_tables,
    generate_role_sql,
)
from promptquery.schema import Column, Schema, Table


def _t(name, *cols):
    return Table("public", name, columns=[Column(c, "text", True, c == "id") for c in cols])


def _schema():
    return Schema(tables=[
        _t("adm_tenants", "id", "slug"),
        _t("sales_orders", "id", "tenant_id", "total_amount"),
        _t("hr_employees", "id", "tenant_id", "first_name", "base_salary", "national_id"),
        _t("core_users", "id", "tenant_id", "password_hash"),
        _t("hr_payroll_run_lines", "id", "tenant_id", "net_amount"),
        _t("core_permissions", "id", "key"),
    ])


def test_denied_tables_get_no_grant_and_no_policy():
    sql = generate_role_sql(_schema())
    for name in ("core_users", "hr_payroll_run_lines", "core_permissions"):
        assert name in DENY_TABLES
        assert f'"{name}"' not in sql


def test_pii_columns_are_excluded_via_column_level_grant():
    sql = generate_role_sql(_schema())
    line = next(
        ln for ln in sql.splitlines()
        if ln.startswith("GRANT SELECT (") and '"hr_employees"' in ln
    )
    assert '"first_name"' in line
    assert "base_salary" not in line and "national_id" not in line
    assert allowed_columns(_schema(), "hr_employees") == ["id", "tenant_id", "first_name"]


def test_every_allowed_tenant_table_has_fail_closed_rls_policy():
    sql = generate_role_sql(_schema())
    assert 'ALTER TABLE "sales_orders" ENABLE ROW LEVEL SECURITY;' in sql
    assert "FORCE" not in sql  # the Atlas owner connection must be unaffected
    # no `, true` missing_ok argument -> unset setting raises instead of returning NULL
    assert "current_setting('app.tenant_id')::uuid" in sql
    assert "current_setting('app.tenant_id', true)" not in sql


def test_tenant_registry_is_scoped_by_its_own_id():
    sql = generate_role_sql(_schema())
    assert 'ON "adm_tenants" FOR SELECT TO atlas_bot USING (id = current_setting' in sql


def test_role_is_read_only_and_limited():
    sql = generate_role_sql(_schema())
    assert "NOSUPERUSER" in sql and "NOCREATEDB" in sql and "NOCREATEROLE" in sql
    assert "default_transaction_read_only = on" in sql
    assert "statement_timeout" in sql
    assert "GRANT INSERT" not in sql and "GRANT UPDATE" not in sql and "GRANT ALL" not in sql


def test_allowed_tables_excludes_deny_and_supports_extra():
    names = allowed_tables(_schema(), extra_deny={"sales_orders"})
    assert "sales_orders" not in names and "adm_tenants" in names
