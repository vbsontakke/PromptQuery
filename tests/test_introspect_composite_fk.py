"""Composite (tenant_id, x_id) foreign keys must not create a tenant_id -> parent edge.

Atlas declares every cross-table FK as (tenant_id, x_id) -> parent(tenant_id, id). Naive
per-column handling yields a `tenant_id` FK to every parent, which makes FK expansion
connect every table to every other table.
"""

from promptquery.schema import introspect


class FakeDB:
    def __init__(self, tables, columns, fks):
        self._rows = {"tables": tables, "columns": columns, "fks": fks}

    def fetch_dicts(self, sql: str):
        if "FROM pg_constraint" in sql:
            return self._rows["fks"]
        if "FROM pg_attribute" in sql:
            return self._rows["columns"]
        return self._rows["tables"]


def _fk(table, col, ref_table, ref_col, constraint, ncols):
    return {
        "schema": "public",
        "table_name": table,
        "column_name": col,
        "referenced_schema": "public",
        "referenced_table": ref_table,
        "referenced_column": ref_col,
        "constraint_name": constraint,
        "ncols": ncols,
    }


def _db(fks):
    tables = [{"schema": "public", "name": n, "comment": None} for n in ("lines", "accounts")]
    cols = [
        {"schema": "public", "table_name": t, "name": c, "data_type": "uuid", "nullable": False,
         "is_primary_key": c == "id", "comment": None, "enum_values": None}
        for t in ("lines", "accounts")
        for c in ("id", "tenant_id", "account_id")
    ]
    return FakeDB(tables, cols, fks)


def test_composite_fk_drops_tenant_column_but_keeps_business_column():
    schema = introspect(_db([
        _fk("lines", "tenant_id", "accounts", "tenant_id", "fk_lines_account", 2),
        _fk("lines", "account_id", "accounts", "id", "fk_lines_account", 2),
    ]))
    lines = next(t for t in schema.tables if t.name == "lines")
    assert [(f.column, f.referenced_table, f.referenced_column) for f in lines.foreign_keys] == [
        ("account_id", "accounts", "id")
    ]


def test_single_column_fk_is_kept_even_for_tenant_id():
    schema = introspect(_db([_fk("lines", "tenant_id", "accounts", "id", "fk_lines_tenant", 1)]))
    lines = next(t for t in schema.tables if t.name == "lines")
    assert len(lines.foreign_keys) == 1


def test_rows_without_constraint_metadata_are_still_accepted():
    row = _fk("lines", "account_id", "accounts", "id", "x", 1)
    del row["constraint_name"], row["ncols"]
    schema = introspect(_db([row]))
    lines = next(t for t in schema.tables if t.name == "lines")
    assert len(lines.foreign_keys) == 1
