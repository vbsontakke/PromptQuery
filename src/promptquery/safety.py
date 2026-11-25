from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import re
import uuid

import sqlglot
from sqlglot import exp
from sqlglot.errors import OptimizeError
from sqlglot.optimizer.qualify import qualify


class UnsafeQuery(Exception):
    """Raised when a generated SQL statement is not a pure read-only SELECT."""


def _node_types(*names: str) -> tuple[type[exp.Expression], ...]:
    # Some node classes only exist in newer sqlglot versions; skip the ones that don't.
    return tuple(getattr(exp, n) for n in names if hasattr(exp, n))


_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = _node_types(
    "Insert",
    "Update",
    "Delete",
    "Merge",
    "Drop",
    "Create",
    "Alter",
    "TruncateTable",
    "Grant",
    "Command",
    # SELECT ... INTO creates a table; FOR UPDATE/SHARE takes row locks.
    "Into",
    "Lock",
    # Session/transaction control could undo the read-only guarantees.
    "Set",
    "SetItem",
    "Transaction",
    "Commit",
    "Rollback",
    "Copy",
)

_ALLOWED_ROOTS: tuple[type[exp.Expression], ...] = (
    exp.Select,
    exp.Union,
    exp.Intersect,
    exp.Except,
    exp.With,
    exp.Subquery,
)

# Postgres functions that can mutate state even from inside a SELECT.
_DANGEROUS_FUNCTIONS = {
    "pg_terminate_backend",
    "pg_cancel_backend",
    "pg_reload_conf",
    "pg_rotate_logfile",
    "lo_import",
    "lo_export",
    "dblink_exec",
    "set_config",
    "current_setting",
    "pg_sleep",
    "pg_read_file",
    "pg_read_binary_file",
    "pg_ls_dir",
    "pg_stat_file",
    "query_to_xml",
    "query_to_xml_and_xmlschema",
    "table_to_xml",
    "cursor_to_xml",
    "nextval",
    "setval",
}

# Anything with these prefixes is a server-admin / file / extension function.
_DANGEROUS_PREFIXES = ("pg_", "lo_", "dblink", "txid_", "xpath_", "query_to_", "table_to_")


def validate_select_only(sql: str) -> None:
    sql = (sql or "").strip()
    if not sql:
        raise UnsafeQuery("empty SQL")

    try:
        statements = sqlglot.parse(sql, read="postgres")
    except Exception as e:
        raise UnsafeQuery(f"could not parse SQL: {e}") from e

    statements = [s for s in statements if s is not None]
    if not statements:
        raise UnsafeQuery("empty SQL")
    if len(statements) > 1:
        raise UnsafeQuery("multiple statements are not allowed")

    stmt = statements[0]

    if not isinstance(stmt, _ALLOWED_ROOTS):
        raise UnsafeQuery(
            f"only SELECT queries are allowed (got {type(stmt).__name__})"
        )

    for forbidden in _FORBIDDEN_NODES:
        node = stmt.find(forbidden)
        if node is not None:
            raise UnsafeQuery(
                f"{type(node).__name__.upper()} statements are not allowed"
            )

    for func in stmt.find_all(exp.Anonymous):
        name = (func.name or "").lower()
        if name in _DANGEROUS_FUNCTIONS:
            raise UnsafeQuery(f"function {name!r} is not allowed")

    for func in stmt.find_all(exp.Func):
        name = (func.sql_name() or "").lower()
        if name in _DANGEROUS_FUNCTIONS:
            raise UnsafeQuery(f"function {name!r} is not allowed")


# ---------------------------------------------------------------------------------------
# Strict validator for the report bot: allowlists, star expansion, LIMIT enforcement.
# ---------------------------------------------------------------------------------------

TENANT_COLUMN = "tenant_id"
_SYNTHETIC_ALIAS = re.compile(r"_col_\d+")
TENANT_ROOT_TABLE = "adm_tenants"  # the tenant registry: its own `id` is the tenant
MAX_SQL_CHARS = 20_000
DEFAULT_ROW_CAP = 1000

# Functions sqlglot does not model (so they parse as Anonymous) that are safe to call.
# Every other anonymous function is rejected, which also rejects all pg_* admin functions.
SAFE_ANONYMOUS_FUNCTIONS: frozenset[str] = frozenset({
    "age",
    "jsonb_extract_path",
    "jsonb_extract_path_text",
    "json_extract_path",
    "json_extract_path_text",
    "jsonb_array_length",
    "json_array_length",
})

_SERIES_FUNCTIONS = {"generate_series", "exploding_generate_series"}


@dataclass(frozen=True)
class ValidatedQuery:
    sql: str  # the SQL to execute: comments stripped, stars expanded, LIMIT enforced
    tables: frozenset[str]  # base tables referenced (CTE names excluded)
    limit_applied: bool  # a LIMIT was injected or clamped


def _is_dangerous_name(name: str) -> bool:
    n = name.lower()
    return n in _DANGEROUS_FUNCTIONS or n.startswith(_DANGEROUS_PREFIXES)


def _check_functions(stmt: exp.Expression, extra_safe: Iterable[str]) -> None:
    safe = SAFE_ANONYMOUS_FUNCTIONS | {f.lower() for f in extra_safe}
    for func in stmt.find_all(exp.Anonymous):
        name = (func.name or "").lower()
        if _is_dangerous_name(name) or name not in safe:
            raise UnsafeQuery(f"function {name!r} is not allowed")
    for func in stmt.find_all(exp.Func):
        if isinstance(func, exp.Anonymous):
            continue
        if _is_dangerous_name(func.sql_name() or ""):
            raise UnsafeQuery(f"function {func.sql_name().lower()!r} is not allowed")


def _check_tables(stmt: exp.Expression, allowed: Mapping[str, Sequence[str]]) -> frozenset[str]:
    cte_names = {c.alias_or_name.lower() for c in stmt.find_all(exp.CTE)}
    seen: set[str] = set()
    for t in stmt.find_all(exp.Table):
        if not t.name:
            # Table-valued function (e.g. generate_series); function checks run separately.
            if isinstance(t.this, exp.Func) and not isinstance(t.this, exp.Anonymous):
                if (t.this.sql_name() or "").lower() not in _SERIES_FUNCTIONS and not isinstance(
                    t.this, exp.Unnest
                ):
                    raise UnsafeQuery("table-valued functions other than generate_series/unnest "
                                      "are not allowed")
            continue
        if t.catalog:
            raise UnsafeQuery("cross-database references are not allowed")
        if t.db and t.db.lower() != "public":
            raise UnsafeQuery(f"schema {t.db!r} is not available")
        name = t.name.lower()
        if not t.db and name in cte_names:
            continue
        if name not in allowed:
            raise UnsafeQuery(f"table {t.name!r} is not available")
        seen.add(name)
    return frozenset(seen)


def _expand_and_validate_columns(
    stmt: exp.Expression, allowed: Mapping[str, Sequence[str]]
) -> exp.Expression:
    """Resolve every column against the readable-column schema.

    Unknown or denied columns raise; ``SELECT *`` is expanded to readable columns only.
    The qualified tree is used for validation always, but is only *executed* when a star had
    to be expanded, because qualification renames unnamed outputs (``COUNT(*)`` -> ``_col_0``).
    """
    schema = {t: {c: "text" for c in cols} for t, cols in allowed.items()}
    has_star = stmt.find(exp.Star) is not None and any(
        isinstance(s, exp.Star) or (isinstance(s, exp.Column) and isinstance(s.this, exp.Star))
        for sel in stmt.find_all(exp.Select)
        for s in sel.expressions
    )
    try:
        qualified = qualify(
            stmt.copy(),
            schema=schema,
            dialect="postgres",
            validate_qualify_columns=True,
            expand_stars=True,
            identify=False,
            quote_identifiers=False,
        )
    except OptimizeError as e:
        raise UnsafeQuery(f"invalid or unavailable column: {e}") from e
    if not has_star:
        return stmt
    # Qualification names unnamed outputs `_col_N`; drop those so Postgres's natural names
    # (count, sum, ...) are preserved for the user.
    def unname(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Alias) and _SYNTHETIC_ALIAS.fullmatch(node.alias or ""):
            return node.this
        return node

    return qualified.transform(unname)


def _apply_row_cap(stmt: exp.Expression, cap: int) -> tuple[exp.Expression, bool]:
    if not isinstance(stmt, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        return stmt, False
    limit = stmt.args.get("limit")
    if limit is None:
        return stmt.limit(cap), True
    if isinstance(limit, exp.Fetch):
        # sqlglot stores `FETCH FIRST n ROWS ONLY` under the limit key as a Fetch node. It is a
        # legitimate limit: keep it when within the cap (replacing it would change the answer),
        # clamp it when above, and leave PERCENT / WITH TIES (no LIMIT equivalent) untouched;
        # execution still caps the rows fetched.
        opts = limit.args.get("limit_options")
        if opts is not None and (opts.args.get("percent") or opts.args.get("with_ties")):
            return stmt, False
        count = limit.args.get("count")
        n = int(count.name) if isinstance(count, exp.Literal) and count.is_int else 1
        if n <= cap:
            return stmt, False
        stmt.set("limit", exp.Limit(expression=exp.Literal.number(cap)))
        return stmt, True
    value = limit.expression
    if isinstance(value, exp.Literal) and value.is_int and int(value.name) <= cap:
        return stmt, False
    stmt.set("limit", exp.Limit(expression=exp.Literal.number(cap)))
    return stmt, True


def scope_to_tenant(
    stmt: exp.Expression,
    tenant_id: str,
    allowed: Mapping[str, Sequence[str]],
) -> exp.Expression:
    """Wrap every tenant-owned table reference in a tenant-filtered subquery.

    ``FROM sales_orders o`` becomes ``FROM (SELECT id, ... FROM sales_orders WHERE tenant_id =
    '<uuid>') AS o``. Because *every* base-table reference is rewritten (joins, subqueries, CTE
    bodies, set operations), a missed predicate is structurally impossible, unlike appending
    ``WHERE tenant_id = ...`` to the outer query. The inner SELECT lists the readable columns
    explicitly so column-level grants (e.g. no salary columns) are never tripped by a ``*``.

    This is defence in depth beside Postgres Row-Level Security: it keeps tenant isolation
    intact even when connected as a role that RLS does not apply to.
    """
    tenant = str(uuid.UUID(str(tenant_id)))
    cte_names = {c.alias_or_name.lower() for c in stmt.find_all(exp.CTE)}

    def swap(node: exp.Expression) -> exp.Expression:
        if not isinstance(node, exp.Table) or node.meta.get("tenant_scoped") or not node.name:
            return node
        name = node.name.lower()
        if (not node.db and name in cte_names) or name not in allowed:
            return node
        cols = list(allowed[name])
        key = "id" if name == TENANT_ROOT_TABLE else TENANT_COLUMN
        if key not in cols:
            return node
        inner_table = exp.to_table(name)
        inner_table.meta["tenant_scoped"] = True
        inner = exp.select(*[exp.column(c) for c in cols]).from_(inner_table)
        inner = inner.where(exp.column(key).eq(exp.Literal.string(tenant)))
        alias = node.alias or node.name
        return exp.Subquery(this=inner, alias=exp.TableAlias(this=exp.to_identifier(alias)))

    return stmt.transform(swap)


def validate_report_query(
    sql: str,
    *,
    allowed_columns: Mapping[str, Sequence[str]],
    row_cap: int = DEFAULT_ROW_CAP,
    extra_safe_functions: Iterable[str] = (),
    tenant_id: str | None = None,
) -> ValidatedQuery:
    """Validate and normalise generated SQL for execution by the report bot.

    ``allowed_columns`` maps each readable table to its readable columns. Anything outside it
    (other tables, other schemas, denied columns) is rejected, as is any function not known to
    be safe. The returned SQL is regenerated from the AST, so comments and other smuggled
    content are gone, and it carries a LIMIT no larger than ``row_cap``.
    """
    sql = (sql or "").strip()
    if len(sql) > MAX_SQL_CHARS:
        raise UnsafeQuery(f"SQL is too long ({len(sql)} > {MAX_SQL_CHARS} characters)")

    validate_select_only(sql)  # single read-only statement, no DML/DDL/INTO/LOCK/SET anywhere
    stmt = sqlglot.parse_one(sql, read="postgres")
    if isinstance(stmt, exp.Subquery):
        stmt = stmt.unnest()

    _check_functions(stmt, extra_safe_functions)
    tables = _check_tables(stmt, {k.lower(): v for k, v in allowed_columns.items()})
    allowed = {k.lower(): v for k, v in allowed_columns.items()}
    stmt = _expand_and_validate_columns(stmt, allowed)
    if tenant_id is not None:
        stmt = scope_to_tenant(stmt, tenant_id, allowed)
    stmt, limited = _apply_row_cap(stmt, row_cap)
    return ValidatedQuery(
        sql=stmt.sql(dialect="postgres", comments=False),
        tables=tables,
        limit_applied=limited,
    )
