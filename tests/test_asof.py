import datetime as dt

from promptquery.atlas.asof import apply_as_of, uses_wall_clock
from promptquery.safety import validate_report_query

AS_OF = dt.date(2026, 7, 6)


def test_current_date_is_replaced_with_literal():
    out = apply_as_of("SELECT CURRENT_DATE", AS_OF)
    assert "2026-07-06" in out and "CURRENT_DATE" not in out.upper()


def test_now_and_current_timestamp_are_replaced():
    for fn in ("now()", "CURRENT_TIMESTAMP", "LOCALTIMESTAMP"):
        out = apply_as_of(f"SELECT {fn}", AS_OF)
        assert "2026-07-06" in out
        assert "NOW" not in out.upper() and "CURRENT_TIMESTAMP" not in out.upper()


def test_date_arithmetic_and_date_trunc_keep_working():
    out = apply_as_of(
        "SELECT * FROM t WHERE d >= date_trunc('month', now()) - INTERVAL '1 month'", AS_OF
    )
    assert "2026-07-06" in out and "INTERVAL" in out.upper()


def test_queries_without_clock_are_equivalent_and_unflagged():
    sql = "SELECT a FROM t WHERE d >= DATE '2026-01-01'"
    assert not uses_wall_clock(sql)
    assert "2026-01-01" in apply_as_of(sql, AS_OF)


def test_unparseable_sql_is_returned_unchanged():
    assert apply_as_of("SELEC FROM WHERE ((", AS_OF) == "SELEC FROM WHERE (("


def test_flags_wall_clock_usage():
    assert uses_wall_clock("SELECT now()")
    assert uses_wall_clock("SELECT CURRENT_DATE")


def test_rewritten_sql_still_passes_the_strict_validator():
    allowed = {"sales_orders": ["id", "order_date", "total_amount"]}
    sql = apply_as_of(
        "SELECT SUM(total_amount) FROM sales_orders WHERE order_date >= CURRENT_DATE - 30", AS_OF
    )
    assert validate_report_query(sql, allowed_columns=allowed).tables == {"sales_orders"}
