import datetime as dt
from decimal import Decimal

import pytest

from promptquery.atlas.pack_loader import Pack, PackColumn, PackTable
from promptquery.db import QueryResult
from promptquery.engine import TURN_LIMIT_MESSAGE, Engine
from promptquery.llm import LLMClient

TENANT = "ab16b03f-aca2-407c-9f5a-839bd09b4e52"
AS_OF = dt.date(2026, 7, 6)


def tiny_pack() -> Pack:
    cols = {
        "sales_orders": ["id", "tenant_id", "order_number", "status", "total_amount", "order_date",
                         "customer_id"],
        "sales_customers": ["id", "tenant_id", "name"],
    }
    tables = {
        n: PackTable(n, "sales", f"{n} desc", False,
                     [PackColumn(c, "text", True, c == "id") for c in cs])
        for n, cs in cols.items()
    }
    return Pack(tables=tables, metrics={}, glossary={}, recipes=[], as_of=AS_OF,
                readable_columns=cols, explicit_joins=[])


def sql_reply(sql, plan="p"):
    return f"<plan>{plan}</plan>\n```sql\n{sql}\n```"


class Scripted(LLMClient):
    name = "scripted"
    model = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []  # (system, messages)

    def generate(self, system, user):
        return self.generate_messages(system, [{"role": "user", "content": user}])

    def generate_messages(self, system, messages, *, cache_system=False, max_tokens=None):
        self.calls.append((system, [dict(m) for m in messages]))
        if not self.replies:
            raise AssertionError("LLM called more times than scripted")
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class FakeDB:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def run_scoped(self, sql, *, tenant_id, max_rows, timeout_ms):
        self.calls.append({"sql": sql, "tenant_id": tenant_id, "max_rows": max_rows})
        r = self.results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def rows(cols, *data, truncated=False):
    return QueryResult(list(cols), [tuple(d) for d in data], truncated)


def engine(llm_replies, db_results, rewrite_replies=(), **kw):
    llm = Scripted(llm_replies)
    fast = Scripted(rewrite_replies)
    db = FakeDB(db_results)
    e = Engine(db=db, pack=tiny_pack(), llm=llm, fast_llm=fast, tenant_id=TENANT, **kw)
    return e, llm, fast, db


# ---------------------------------------------------------------------------------------

def test_happy_path_validates_scopes_and_executes():
    e, llm, _, db = engine(
        [sql_reply("SELECT status, COUNT(*) AS n FROM sales_orders GROUP BY status")],
        [rows(["status", "n"], ("CONFIRMED", 19), ("CLOSED", 66))],
    )
    r = e.ask("s1", "How many orders per status?")
    assert r.status == "ok" and r.turn_no == 1 and r.row_count == 2
    assert r.columns == ["status", "n"] and r.rows[0] == ("CONFIRMED", 19)
    call = db.calls[0]
    assert call["tenant_id"] == TENANT and call["max_rows"] == 1000
    assert f"tenant_id = '{TENANT}'" in call["sql"] and "LIMIT 1000" in call["sql"]
    assert llm.calls[0][1][0]["content"].endswith("How many orders per status?")
    assert "2026-07-06" in llm.calls[0][0]  # the as-of date is in the system prompt


def test_to_dict_is_json_serialisable():
    import json

    e, *_ = engine([sql_reply("SELECT total_amount, order_date FROM sales_orders")],
                   [rows(["total_amount", "order_date"], (Decimal("12.50"), dt.date(2026, 4, 1)))])
    d = e.ask("s", "orders").to_dict()
    assert json.loads(json.dumps(d))["rows"] == [["12.50", "2026-04-01"]]


def test_refusal_never_touches_the_database():
    e, _, _, db = engine(["REFUSE: salary data is not available"], [])
    r = e.ask("s", "show everyone's salary")
    assert r.status == "refused" and "salary" in r.answer_text and db.calls == []


def test_validator_rejection_triggers_repair_with_feedback():
    e, llm, _, db = engine(
        [sql_reply("SELECT * FROM core_users"), sql_reply("SELECT COUNT(*) FROM sales_orders")],
        [rows(["count"], (91,))],
    )
    r = e.ask("s", "q")
    assert r.status == "ok" and len(db.calls) == 1  # the rejected SQL never reached the DB
    feedback = llm.calls[1][1][-1]["content"]
    assert "rejected by the safety validator" in feedback and "core_users" in feedback
    assert llm.calls[1][1][-2]["role"] == "assistant"
    assert any("validator rejected" in n for n in r.notes)


def test_database_error_triggers_repair():
    e, llm, _, _ = engine(
        [sql_reply("SELECT COUNT(*) FROM sales_orders"), sql_reply("SELECT COUNT(id) FROM sales_orders")],
        [RuntimeError('column "x" does not exist\nLINE 1'), rows(["count"], (91,))],
    )
    r = e.ask("s", "q")
    assert r.status == "ok" and 'column "x" does not exist' in llm.calls[1][1][-1]["content"]


def test_gives_up_after_max_repairs_and_reports_error_but_counts_the_turn():
    e, llm, _, _ = engine([sql_reply("SELECT * FROM core_users")] * 3, [], max_repairs=2)
    r = e.ask("s", "q")
    assert r.status == "error" and "safety validator" in r.error and len(llm.calls) == 3
    assert e.store.get("s").turn_count == 1


def test_empty_result_is_double_checked_once_and_a_better_query_wins():
    e, llm, _, db = engine(
        [sql_reply("SELECT id FROM sales_orders WHERE status = 'OPEN'"),
         sql_reply("SELECT id FROM sales_orders WHERE status = 'CONFIRMED'")],
        [rows(["id"]), rows(["id"], ("a",), ("b",))],
    )
    r = e.ask("s", "open orders")
    assert r.status == "ok" and r.row_count == 2 and len(db.calls) == 2
    assert "returned no rows" in llm.calls[1][1][-1]["content"] and "2026-07-06" in llm.calls[1][1][-1]["content"]


def test_genuinely_empty_result_is_accepted_after_one_recheck():
    same = sql_reply("SELECT id FROM sales_orders WHERE status = 'CANCELLED'")
    e, _, _, db = engine([same, same], [rows(["id"]), rows(["id"])])
    r = e.ask("s", "cancelled orders")
    assert r.status == "ok" and r.row_count == 0 and r.answer_text == "No matching records."
    assert len(db.calls) == 2


def test_empty_retry_that_errors_falls_back_to_the_empty_result():
    e, _, _, _ = engine(
        [sql_reply("SELECT id FROM sales_orders WHERE status = 'X'"), sql_reply("SELECT * FROM core_users"),
         sql_reply("SELECT * FROM core_users")],
        [rows(["id"])],
    )
    r = e.ask("s", "q")
    assert r.status == "ok" and r.row_count == 0 and any("kept the original empty" in n for n in r.notes)


def test_wall_clock_functions_are_rewritten_to_the_as_of_date():
    e, _, _, db = engine(
        [sql_reply("SELECT COUNT(*) FROM sales_orders WHERE order_date >= CURRENT_DATE - 30")],
        [rows(["count"], (3,))],
    )
    r = e.ask("s", "orders in the last 30 days")
    assert "2026-07-06" in db.calls[0]["sql"] and "CURRENT_DATE" not in db.calls[0]["sql"].upper()
    assert any("as-of" in n for n in r.notes)


def test_reply_without_sql_is_repaired():
    e, llm, _, _ = engine(["Sure! Let me think about that.", sql_reply("SELECT COUNT(*) FROM sales_orders")],
                          [rows(["count"], (1,))])
    assert e.ask("s", "q").status == "ok"
    assert "could not find a SQL block" in llm.calls[1][1][-1]["content"]


def test_llm_outage_is_an_error_result_not_an_exception():
    e, _, _, _ = engine([RuntimeError("503 overloaded")], [])
    r = e.ask("s", "q")
    assert r.status == "error" and "unavailable" in r.error


def test_empty_question_is_an_error_without_calling_anything():
    e, llm, _, _ = engine([], [])
    assert e.ask("s", "   ").status == "error" and llm.calls == []


# ---- multi-turn ---------------------------------------------------------------------

def test_followup_refine_gets_base_sql_history_and_rewrite():
    e, llm, fast, db = engine(
        [sql_reply("SELECT c.name, SUM(o.total_amount) AS rev FROM sales_orders o "
                   "JOIN sales_customers c ON c.id = o.customer_id GROUP BY c.name ORDER BY rev DESC"),
         sql_reply("SELECT c.name, SUM(o.total_amount) AS rev FROM sales_orders o "
                   "JOIN sales_customers c ON c.id = o.customer_id GROUP BY c.name ORDER BY rev DESC LIMIT 5")],
        [rows(["name", "rev"], ("Northwind", Decimal("2950")), ("Contoso", Decimal("2750"))),
         rows(["name", "rev"], ("Northwind", Decimal("2950")))],
        rewrite_replies=['{"intent":"REFINE","standalone_question":"Top 5 customers by revenue","base_turn":1}'],
    )
    e.ask("s", "revenue by customer")
    r = e.ask("s", "only the top 5")
    assert r.intent == "REFINE" and r.standalone_question == "Top 5 customers by revenue"
    msg = llm.calls[1][1][0]["content"]
    assert "builds on turn 1" in msg and "GROUP BY c.name" in msg
    assert '<prior_result untrusted="true">' in msg and "Northwind" in msg
    assert len(fast.calls) == 1  # turn 1 never calls the rewriter


def test_ordinal_reference_is_resolved_and_passed_through():
    e, llm, fast, _ = engine(
        [sql_reply("SELECT c.name, SUM(o.total_amount) AS rev FROM sales_orders o JOIN sales_customers c ON c.id = o.customer_id GROUP BY c.name"),
         sql_reply("SELECT order_number FROM sales_orders o JOIN sales_customers c ON c.id = o.customer_id WHERE c.name = 'Northwind'")],
        [rows(["name", "rev"], ("Northwind", Decimal("2950")), ("Contoso", Decimal("2750"))),
         rows(["order_number"], ("SO-1",))],
        rewrite_replies=['{"intent":"DRILL","standalone_question":"Orders of customer Northwind","base_turn":1}'],
    )
    e.ask("s", "revenue by customer")
    e.ask("s", "show the orders of the first one")
    assert "name=Northwind" in fast.calls[0][1][0]["content"]
    assert "Reference resolved" in llm.calls[1][1][0]["content"]


def test_clarify_intent_asks_without_running_sql_and_counts_a_turn():
    e, llm, _, db = engine(
        [sql_reply("SELECT COUNT(*) FROM sales_orders")], [rows(["count"], (1,))],
        rewrite_replies=['{"intent":"CLARIFY","clarification":"Which customer do you mean?"}'],
    )
    e.ask("s", "orders")
    r = e.ask("s", "show that one")
    assert r.status == "clarify" and "Which customer" in r.answer_text
    assert len(db.calls) == 1 and len(llm.calls) == 1 and e.store.get("s").turn_count == 2


def test_reset_intent_ignores_history_in_the_generation_prompt():
    e, llm, _, _ = engine(
        [sql_reply("SELECT COUNT(*) FROM sales_orders"), sql_reply("SELECT COUNT(*) FROM sales_customers")],
        [rows(["count"], (1,)), rows(["count"], (2,))],
        rewrite_replies=['{"intent":"RESET","standalone_question":"How many customers?"}'],
    )
    e.ask("s", "how many orders")
    e.ask("s", "never mind, how many customers?")
    assert "Conversation so far" not in llm.calls[1][1][0]["content"]


def test_six_turns_then_turn_limit_without_any_model_or_db_call():
    e, llm, fast, db = engine(
        [sql_reply("SELECT COUNT(*) FROM sales_orders")] * 6, [rows(["count"], (1,))] * 6,
        rewrite_replies=['{"intent":"NEW","standalone_question":"x"}'] * 5,
    )
    results = [e.ask("s", f"question {i}") for i in range(6)]
    assert [r.turn_no for r in results] == [1, 2, 3, 4, 5, 6] and all(r.status == "ok" for r in results)
    calls = (len(llm.calls), len(fast.calls), len(db.calls))
    r7 = e.ask("s", "one more")
    assert r7.status == "turn_limit" and r7.error == "TURN_LIMIT" and r7.answer_text == TURN_LIMIT_MESSAGE
    assert (len(llm.calls), len(fast.calls), len(db.calls)) == calls
    assert e.store.get("s").turn_count == 6  # state retained, not extended


def test_sessions_are_isolated_from_each_other():
    e, llm, fast, _ = engine(
        [sql_reply("SELECT COUNT(*) FROM sales_orders")] * 2, [rows(["count"], (1,))] * 2)
    e.ask("a", "q1")
    e.ask("b", "q2")
    assert len(fast.calls) == 0  # both were first turns of their own session
    assert e.store.get("a").turn_count == 1 and e.store.get("b").turn_count == 1


def test_reset_session_starts_over():
    e, *_ = engine([sql_reply("SELECT COUNT(*) FROM sales_orders")], [rows(["count"], (1,))])
    e.ask("s", "q")
    e.reset_session("s")
    assert e.store.get("s") is None


def test_audit_log_records_the_attempts(tmp_path):
    import json

    from promptquery.audit import AuditLog

    path = tmp_path / "a.jsonl"
    e, *_ = engine(
        [sql_reply("SELECT * FROM core_users"), sql_reply("SELECT COUNT(*) FROM sales_orders")],
        [rows(["count"], (1,))], audit=AuditLog(path),
    )
    e.ask("s", "q")
    events = [json.loads(x)["event"] for x in path.read_text().splitlines()]
    assert events.count("generated") == 2 and "rejected" in events and events[-1] == "turn_done"


@pytest.mark.parametrize("bad", ["DROP TABLE sales_orders", "DELETE FROM sales_orders",
                                 "SELECT pg_sleep(100)", "SELECT 1; SELECT 2"])
def test_dangerous_model_output_never_reaches_the_database(bad):
    e, _, _, db = engine([sql_reply(bad)] * 3, [], max_repairs=2)
    assert e.ask("s", "q").status == "error" and db.calls == []
