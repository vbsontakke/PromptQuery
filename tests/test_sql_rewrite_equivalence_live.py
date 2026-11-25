"""The validator regenerates SQL (star expansion, tenant scoping, LIMIT, comment stripping).
A false rejection or a changed result would silently cost accuracy, so a battery of realistic
model-style PostgreSQL queries must (a) pass validation and (b) return exactly the same rows
after rewriting as the original SQL does when run directly under Row-Level Security."""

import os
from collections import Counter

import pytest

from eval.atlas.compare import norm
from promptquery.bootstrap import Settings, build_engine
from promptquery.safety import validate_report_query
from tests.test_engine import Scripted

OWNER = os.environ.get("PQ_TEST_OWNER_DSN")
BOT = os.environ.get("PQ_TEST_BOT_DSN")

pytestmark = pytest.mark.skipif(not (OWNER and BOT), reason="needs a seeded Atlas DB")

# (tenant, sql). Stay under the 1000-row cap so results are comparable.
BATTERY = [
    ("acme", "SELECT status, COUNT(*) FROM sales_orders GROUP BY status"),
    ("acme", "SELECT COUNT(*) AS n, COALESCE(SUM(total_amount), 0) AS v FROM sales_orders WHERE status IN ('CONFIRMED','PARTIALLY_DELIVERED')"),
    ("acme", "SELECT * FROM fin_accounts ORDER BY code"),
    ("acme", "SELECT a.* FROM fin_accounts a WHERE a.account_type = 'EXPENSE'"),
    ("acme", "SELECT o.*, c.name FROM sales_orders o JOIN sales_customers c ON c.id = o.customer_id"),
    ("acme", "SELECT c.name, SUM(o.total_amount) FILTER (WHERE o.status = 'CLOSED') AS closed_value FROM sales_orders o JOIN sales_customers c ON c.id = o.customer_id GROUP BY c.name"),
    ("acme", "SELECT DISTINCT ON (customer_id) customer_id, order_number, order_date FROM sales_orders ORDER BY customer_id, order_date DESC"),
    ("acme", "SELECT order_number, total_amount, RANK() OVER (ORDER BY total_amount DESC) AS rnk FROM sales_orders"),
    ("acme", "SELECT order_number, SUM(total_amount) OVER (ORDER BY order_date, order_number) AS running FROM sales_orders"),
    ("acme", "SELECT customer_id, order_date, LAG(order_date) OVER (PARTITION BY customer_id ORDER BY order_date) AS prev FROM sales_orders"),
    ("acme", "SELECT order_number, total_amount::numeric(12,2) AS amt, order_date::date FROM sales_orders"),
    ("acme", "SELECT order_number, order_date + INTERVAL '30 days' AS due FROM sales_orders"),
    ("acme", "SELECT order_number FROM sales_orders WHERE order_date >= DATE '2026-05-01' - INTERVAL '1 month'"),
    ("acme", "SELECT to_char(order_date, 'YYYY-MM') AS ym, COUNT(*) FROM sales_orders GROUP BY 1 ORDER BY 1"),
    ("acme", "SELECT EXTRACT(MONTH FROM order_date) AS m, COUNT(*) FROM sales_orders GROUP BY 1"),
    ("acme", "SELECT date_trunc('month', order_date)::date AS m, AVG(total_amount) FROM sales_orders GROUP BY 1"),
    ("acme", "SELECT name FROM sales_customers WHERE name ILIKE '%north%'"),
    ("acme", "SELECT name FROM sales_customers WHERE name LIKE 'C%' OR name LIKE 'F%'"),
    ("acme", "SELECT CASE WHEN total_amount > 800 THEN 'large' WHEN total_amount > 500 THEN 'medium' ELSE 'small' END AS size, COUNT(*) FROM sales_orders GROUP BY 1"),
    ("acme", "SELECT COALESCE(SUM(open_amount), 0) FROM fin_customer_invoices WHERE due_date < DATE '2026-07-06'"),
    ("acme", "SELECT customer_id, COUNT(*) FROM sales_orders GROUP BY customer_id HAVING COUNT(*) > 3"),
    ("acme", "SELECT order_number FROM sales_orders WHERE total_amount BETWEEN 500 AND 900 ORDER BY total_amount DESC NULLS LAST"),
    ("acme", "SELECT order_number FROM sales_orders WHERE customer_id IN (SELECT id FROM sales_customers WHERE status = 'ACTIVE')"),
    ("acme", "SELECT c.name FROM sales_customers c WHERE EXISTS (SELECT 1 FROM sales_orders o WHERE o.customer_id = c.id AND o.status = 'CONFIRMED')"),
    ("acme", "SELECT c.name FROM sales_customers c WHERE NOT EXISTS (SELECT 1 FROM fin_vendor_bills b WHERE b.partner_id = c.id)"),
    ("acme", "WITH m AS (SELECT customer_id, SUM(total_amount) AS v FROM sales_orders GROUP BY customer_id) SELECT c.name, m.v FROM m JOIN sales_customers c ON c.id = m.customer_id ORDER BY m.v DESC"),
    ("acme", "WITH a AS (SELECT id FROM sales_orders), b AS (SELECT id FROM a) SELECT COUNT(*) FROM b"),
    ("acme", "SELECT name FROM sales_customers UNION SELECT name FROM proc_vendors ORDER BY 1"),
    ("acme", "SELECT id FROM sales_customers INTERSECT SELECT customer_id FROM sales_orders"),
    ("acme", "SELECT string_agg(name, ', ' ORDER BY name) FROM sales_customers"),
    ("acme", "SELECT array_agg(DISTINCT status) FROM sales_orders"),
    ("acme", "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY total_amount) FROM sales_orders"),
    ("acme", "SELECT ROUND(AVG(total_amount)::numeric, 2), MIN(total_amount), MAX(total_amount), STDDEV(total_amount) FROM sales_orders"),
    ("acme", "SELECT COUNT(DISTINCT customer_id), COUNT(*) FILTER (WHERE status = 'CLOSED') FROM sales_orders"),
    ("acme", "SELECT g.d::date FROM generate_series(DATE '2026-04-01', DATE '2026-04-05', INTERVAL '1 day') AS g(d)"),
    ("acme", "SELECT d.m, COALESCE(x.n, 0) FROM (SELECT generate_series(1, 6) AS m) d LEFT JOIN (SELECT EXTRACT(MONTH FROM order_date)::int AS m, COUNT(*) AS n FROM sales_orders GROUP BY 1) x ON x.m = d.m"),
    ("acme", "SELECT order_number, total_amount FROM sales_orders ORDER BY total_amount DESC FETCH FIRST 3 ROWS ONLY"),
    ("acme", "SELECT order_number FROM sales_orders ORDER BY order_date DESC LIMIT 3 OFFSET 1"),
    ("acme", "SELECT age(DATE '2026-07-06', order_date) FROM sales_orders"),
    ("acme", "SELECT a.code, SUM(l.functional_debit_amount) - SUM(l.functional_credit_amount) AS bal FROM fin_journal_lines l JOIN fin_accounts a ON a.id = l.account_id WHERE l.is_posted GROUP BY a.code HAVING SUM(l.functional_debit_amount) <> SUM(l.functional_credit_amount) ORDER BY a.code"),
    ("acme", "SELECT i.item_code, COALESCE(q.on_hand, 0) FROM inv_items i LEFT JOIN (SELECT item_id, SUM(on_hand_qty) AS on_hand FROM inv_stock_quants GROUP BY item_id) q ON q.item_id = i.id"),
    ("acme", "SELECT e.first_name, m.first_name AS manager FROM hr_employees e LEFT JOIN hr_employees m ON m.id = e.manager_id"),
    ("acme", "SELECT e.first_name, e.last_name FROM hr_employees e"),
    ("acme", "SELECT COUNT(*) FROM (SELECT DISTINCT customer_id FROM sales_orders) t"),
    ("acme", "SELECT status, ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct FROM sales_orders GROUP BY status"),
    ("acme", "SELECT o.order_number, (SELECT COUNT(*) FROM sales_order_lines l WHERE l.order_id = o.id) AS lines FROM sales_orders o"),
    ("acme", "SELECT order_number, NULLIF(requested_date - order_date, 0) AS days FROM sales_orders"),
    ("acme", "SELECT COUNT(*) FROM sales_orders WHERE requested_date IS NOT NULL AND requested_date IS DISTINCT FROM order_date"),
    ("acme", "SELECT LEAST(total_amount, 700) AS capped, GREATEST(total_amount, 100) FROM sales_orders"),
    ("acme", "SELECT UPPER(name), LENGTH(name), SUBSTRING(name FROM 1 FOR 3), name || ' (' || customer_code || ')' FROM sales_customers"),
    ("manufacturing", "SELECT v.name, COUNT(*) FROM proc_purchase_orders p JOIN proc_vendors v ON v.id = p.vendor_id GROUP BY v.name"),
    ("hospitality", "SELECT t.status, SUM(l.line_amount) FROM hsp_order_tickets t JOIN hsp_order_ticket_lines l ON l.ticket_id = t.id GROUP BY t.status"),
    ("volume", "SELECT EXTRACT(YEAR FROM order_date)::int AS y, COUNT(*), SUM(total_amount) FROM sales_orders GROUP BY 1 ORDER BY 1"),
    ("volume", "SELECT c.name, SUM(o.total_amount) AS v, RANK() OVER (ORDER BY SUM(o.total_amount) DESC) AS r FROM sales_orders o JOIN sales_customers c ON c.id = o.customer_id GROUP BY c.name ORDER BY v DESC LIMIT 10"),
    ("volume", "SELECT i.item_code, v.total_value FROM inv_item_valuations v JOIN inv_items i ON i.id = v.item_id ORDER BY v.total_value DESC, i.item_code LIMIT 25"),
    ("volume", "SELECT date_trunc('quarter', move_date)::date AS q, move_type, SUM(quantity) FROM inv_stock_moves GROUP BY 1, 2 ORDER BY 1, 2"),
]


@pytest.fixture(scope="module")
def engines():
    cache = {}

    def get(tenant):
        if tenant not in cache:
            cache[tenant] = build_engine(
                Settings(dsn=BOT, tenant=tenant, admin_dsn=OWNER), llm=Scripted([]), fast_llm=Scripted([])
            )
        return cache[tenant]

    return get


def _rows(rows):
    return Counter(tuple(round(c, 4) if isinstance(c, float) else c for c in (norm(v) for v in r)) for r in rows)


@pytest.mark.parametrize("tenant,sql", BATTERY, ids=[f"{i:02d}" for i in range(len(BATTERY))])
def test_validated_rewrite_returns_the_same_rows_as_the_original(engines, tenant, sql):
    eng = engines(tenant)
    original = eng.db.run_scoped(sql, tenant_id=eng.tenant_id, max_rows=100_000)  # RLS only
    vq = validate_report_query(sql, allowed_columns=eng.pack.allowed_columns(), row_cap=100_000,
                               tenant_id=eng.tenant_id)
    rewritten = eng.db.run_scoped(vq.sql, tenant_id=eng.tenant_id, max_rows=100_000)
    assert len(rewritten.columns) == len(original.columns), (original.columns, rewritten.columns)
    assert rewritten.columns == original.columns, "output column names must be preserved"
    assert _rows(rewritten.rows) == _rows(original.rows)
