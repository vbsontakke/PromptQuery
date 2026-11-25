import json
import threading
from decimal import Decimal

from promptquery.audit import AuditLog


def test_records_are_appended_as_jsonl(tmp_path):
    log = AuditLog(tmp_path / "sub" / "audit.jsonl")
    log.record("attempt", session_id="s1", turn_no=1, sql="SELECT 1", total=Decimal("1.5"))
    log.record("attempt", session_id="s1", turn_no=2, error="boom")
    lines = (tmp_path / "sub" / "audit.jsonl").read_text().splitlines()
    rows = [json.loads(x) for x in lines]
    assert [r["turn_no"] for r in rows] == [1, 2]
    assert rows[0]["total"] == "1.5" and rows[1]["error"] == "boom"
    assert all("ts" in r and r["event"] == "attempt" for r in rows)


def test_disabled_log_is_a_noop():
    assert AuditLog(None).record("x", a=1)["a"] == 1


def test_concurrent_writes_stay_one_record_per_line(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl")
    threads = [threading.Thread(target=lambda i=i: log.record("e", i=i)) for i in range(40)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    rows = [json.loads(x) for x in (tmp_path / "a.jsonl").read_text().splitlines()]
    assert sorted(r["i"] for r in rows) == list(range(40))
