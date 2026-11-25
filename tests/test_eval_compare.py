import datetime as dt
import uuid
from decimal import Decimal

from eval.atlas.compare import CompareOptions, compare_results

D = Decimal


def ok(g, c, **kw):
    return compare_results(g, c, CompareOptions(**kw))


def test_identical_results_match():
    assert ok([("A", D("1.5"))], [("A", D("1.5"))]).ok


def test_column_names_and_order_do_not_matter():
    assert ok([("A", D("10"))], [(D("10"), "A")]).ok


def test_row_order_ignored_by_default_but_enforced_when_requested():
    g, c = [("A", 3), ("B", 2)], [("B", 2), ("A", 3)]
    assert ok(g, c).ok
    assert not ok(g, c, order=True).ok
    assert ok(g, g, order=True).ok


def test_numeric_tolerance_int_vs_decimal_vs_float():
    assert ok([(D("100.000000"),)], [(100,)]).ok
    assert ok([(D("33.333333"),)], [(33.33,)]).ok
    assert not ok([(D("100"),)], [(101,)]).ok


def test_wrong_value_is_caught():
    r = ok([("A", D("10"))], [("A", D("11"))])
    assert not r.ok and r.reason


def test_row_count_mismatch_in_either_direction():
    assert not ok([(1,), (2,)], [(1,)]).ok
    assert not ok([(1,)], [(1,), (2,)]).ok


def test_duplicates_matter():
    assert not ok([(1,), (1,), (2,)], [(1,), (2,), (2,)]).ok


def test_extra_identifying_columns_allowed_by_default_not_when_strict():
    g, c = [(D("10"),)], [("Northwind", D("10"))]
    assert ok(g, c).ok
    assert not ok(g, c, extra_columns=False).ok


def test_missing_column_fails():
    assert not ok([("A", 1)], [("A",)]).ok


def test_empty_results():
    assert ok([], []).ok
    assert not ok([], [(1,)]).ok
    assert not ok([(1,)], []).ok


def test_null_handling():
    assert ok([(None, 1)], [(None, 1)]).ok
    assert not ok([(None,)], [(0,)]).ok


def test_dates_uuids_strings_canonicalised():
    u = uuid.uuid4()
    assert ok([(dt.date(2026, 4, 1), str(u))], [(dt.datetime(2026, 4, 1), u)]).ok is False  # datetime != date
    assert ok([(dt.date(2026, 4, 1), str(u), " x ")], [("2026-04-01", u, "x")]).ok


def test_swapped_columns_with_same_values_cannot_pass_as_correct_pairing():
    # gold pairs (A,1),(B,2); candidate pairs (A,2),(B,1): same column multisets, wrong rows
    assert not ok([("A", 1), ("B", 2)], [("A", 2), ("B", 1)]).ok


def test_ranking_with_ties_needs_order_false_to_be_robust():
    g = [("A", 5), ("B", 5)]
    assert ok(g, [("B", 5), ("A", 5)]).ok
