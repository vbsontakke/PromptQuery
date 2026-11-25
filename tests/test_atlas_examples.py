import os
import re

import pytest

from eval.atlas.scenario import load_scenarios
from promptquery.atlas.examples import Example, ExampleLibrary, render_examples
from promptquery.atlas.prompts import build_system_prompt, build_user_message
from promptquery.bootstrap import Settings, build_engine
from promptquery.db import QueryResult
from promptquery.engine import Engine
from promptquery.retrieval import tokenize
from promptquery.safety import validate_report_query
from tests.test_engine import TENANT, FakeDB, Scripted, sql_reply, tiny_pack

OWNER = os.environ.get("PQ_TEST_OWNER_DSN")
BOT = os.environ.get("PQ_TEST_BOT_DSN")


def _norm_sql(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().rstrip(";")).lower()


# ---- retrieval -----------------------------------------------------------------------

def lib():
    return ExampleLibrary([
        Example("How many orders did each customer place in April?", sql="SELECT 1"),
        Example("What is the total value of purchase orders per month?", sql="SELECT 2"),
        Example("Which employees earn more than 50000?", refuse="Salary data is not available."),
    ])


def test_most_similar_example_ranks_first():
    r = lib().retrieve("How many orders did every customer place in March?", k=3)
    assert r[0].sql == "SELECT 1"


def test_refusal_examples_are_retrievable_and_rendered_as_refusals():
    r = lib().retrieve("Which employees make more than 90000 a year?", k=1)
    assert r and r[0].refuse
    assert "REFUSE: Salary data is not available." in render_examples(r)


def test_unrelated_question_gets_no_examples():
    assert lib().retrieve("zebra quantum helicopter", k=3) == []
    assert lib().retrieve("", k=3) == []
    assert render_examples([]) == ""


def test_default_library_loads_with_sql_and_refusals():
    L = ExampleLibrary.load()
    assert len(L.examples) >= 40
    assert sum(1 for e in L.examples if e.refuse) >= 3
    assert all((e.sql is None) != (e.refuse is None) for e in L.examples)


def test_examples_are_rendered_into_the_user_message():
    text = build_user_message(question="q", history=None, base_sql=None, base_turn=None,
                              ordinal_hint=None, examples=render_examples(lib().retrieve("orders per customer")))
    assert "Verified examples" in text and "Question to answer now: q" in text


def test_engine_shows_retrieved_examples_to_the_model():
    llm = Scripted([sql_reply("SELECT COUNT(*) FROM sales_orders")])
    e = Engine(db=FakeDB([QueryResult(["count"], [(1,)], False)]), pack=tiny_pack(), llm=llm,
               fast_llm=Scripted([]), tenant_id=TENANT, examples=lib())
    e.ask("s", "How many orders did each customer place in June?")
    sent = llm.calls[0][1][0]["content"]
    assert "Verified examples" in sent and "SELECT 1" in sent


def test_engine_without_a_library_sends_no_examples_block():
    llm = Scripted([sql_reply("SELECT COUNT(*) FROM sales_orders")])
    e = Engine(db=FakeDB([QueryResult(["count"], [(1,)], False)]), pack=tiny_pack(), llm=llm,
               fast_llm=Scripted([]), tenant_id=TENANT)
    e.ask("s", "orders")
    assert "Verified examples" not in llm.calls[0][1][0]["content"]


# ---- the system prompt ---------------------------------------------------------------

def test_system_prompt_renders_with_no_unresolved_placeholders():
    p = build_system_prompt(tiny_pack())
    assert not re.findall(r"\{(?:as_of|schema|joins|glossary|metrics)\}", p)
    assert "2026-07-06" in p and "{braces}" in p and "REFUSE:" in p and "<plan>" in p
    assert "# Schema" in p and "sales_orders" in p


# ---- eval integrity: examples must never duplicate evaluation scenarios --------------

def test_examples_are_disjoint_from_the_evaluation_set():
    eval_qs = [(t.user, _norm_sql(t.gold_sql or "")) for s in load_scenarios() for t in s.turns]
    eval_tokens = [(set(tokenize(q)), q) for q, _ in eval_qs]
    eval_sql = {sql for _, sql in eval_qs if sql}
    bad = []
    for ex in ExampleLibrary.load().examples:
        toks = set(tokenize(ex.question))
        for et, q in eval_tokens:
            jac = len(toks & et) / max(len(toks | et), 1)
            if jac >= 0.7:
                bad.append(f"{ex.question!r} ~ eval {q!r} (jaccard {jac:.2f})")
        if ex.sql and _norm_sql(ex.sql) in eval_sql:
            bad.append(f"{ex.question!r}: SQL identical to an eval gold")
    assert not bad, "\n".join(bad)


def test_example_questions_are_unique():
    qs = [e.question.lower() for e in ExampleLibrary.load().examples]
    assert len(qs) == len(set(qs))


@pytest.mark.skipif(not (OWNER and BOT), reason="needs a seeded Atlas DB")
def test_every_example_validates_and_returns_rows_on_real_data():
    engines: dict[str, Engine] = {}
    problems = []
    for ex in ExampleLibrary.load().examples:
        if not ex.sql:
            continue
        eng = engines.get(ex.tenant) or build_engine(
            Settings(dsn=BOT, tenant=ex.tenant, admin_dsn=OWNER), llm=Scripted([]), fast_llm=Scripted([])
        )
        engines[ex.tenant] = eng
        try:
            vq = validate_report_query(ex.sql, allowed_columns=eng.pack.allowed_columns(),
                                       row_cap=100_000, tenant_id=eng.tenant_id)
            res = eng.db.run_scoped(vq.sql, tenant_id=eng.tenant_id, max_rows=100_000)
        except Exception as err:  # noqa: BLE001
            problems.append(f"{ex.question!r}: {str(err).splitlines()[0]}")
            continue
        if not res.rows:
            problems.append(f"{ex.question!r}: returns no rows")
    assert not problems, "\n".join(problems)
