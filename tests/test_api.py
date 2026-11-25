from fastapi.testclient import TestClient

from promptquery.api.app import create_app
from promptquery.db import QueryResult
from promptquery.engine import Engine
from tests.test_engine import AS_OF, TENANT, FakeDB, Scripted, sql_reply, tiny_pack  # noqa: F401


def client(n_replies=8):
    llm = Scripted([sql_reply("SELECT COUNT(*) FROM sales_orders")] * n_replies)
    fast = Scripted(['{"intent":"NEW","standalone_question":"x"}'] * n_replies)
    db = FakeDB([QueryResult(["count"], [(91,)], False)] * n_replies)
    eng = Engine(db=db, pack=tiny_pack(), llm=llm, fast_llm=fast, tenant_id=TENANT)
    return TestClient(create_app(eng)), eng


def test_health():
    c, _ = client()
    assert c.get("/healthz").json() == {"status": "ok"}


def test_chat_returns_a_turn_result_and_creates_the_session():
    c, _ = client()
    r = c.post("/chat", json={"message": "how many orders?", "session_id": "abc"})
    body = r.json()
    assert r.status_code == 200 and body["status"] == "ok" and body["turn_no"] == 1
    assert body["rows"] == [[91]] and body["columns"] == ["count"] and body["session_id"] == "abc"
    assert body["sql"].upper().startswith("SELECT")


def test_chat_without_session_id_gets_a_fresh_one():
    c, _ = client()
    a = c.post("/chat", json={"message": "q"}).json()["session_id"]
    b = c.post("/chat", json={"message": "q"}).json()["session_id"]
    assert a and b and a != b


def test_seventh_message_returns_turn_limit_over_http():
    c, _ = client()
    sid = c.post("/sessions").json()["session_id"]
    for i in range(6):
        assert c.post("/chat", json={"message": f"q{i}", "session_id": sid}).json()["status"] == "ok"
    r = c.post("/chat", json={"message": "q7", "session_id": sid}).json()
    assert r["status"] == "turn_limit" and r["error"] == "TURN_LIMIT"
    info = c.get(f"/sessions/{sid}").json()
    assert info["turns"] == 6 and info["max_turns"] == 6 and len(info["history"]) == 6


def test_delete_session_resets_the_conversation():
    c, _ = client()
    sid = "reset-me"
    for i in range(6):
        c.post("/chat", json={"message": f"q{i}", "session_id": sid})
    assert c.delete(f"/sessions/{sid}").json() == {"deleted": True}
    assert c.get(f"/sessions/{sid}").status_code == 404
    assert c.post("/chat", json={"message": "again", "session_id": sid}).json()["turn_no"] == 1


def test_input_validation():
    c, _ = client()
    assert c.post("/chat", json={"message": ""}).status_code == 422
    assert c.post("/chat", json={"message": "x" * 2001}).status_code == 422
    assert c.post("/chat", json={"message": "q", "session_id": "bad id!"}).status_code == 422
    assert c.post("/chat", json={}).status_code == 422
    assert c.get("/sessions/unknown").status_code == 404


def test_api_key_protects_every_route_but_health():
    from promptquery.api.app import create_app

    c, eng = client()
    c = TestClient(create_app(eng, api_key="s3cret"))
    assert c.get("/healthz").status_code == 200
    for method, path, body in [("post", "/chat", {"message": "q"}), ("post", "/sessions", None),
                               ("get", "/sessions/x", None), ("delete", "/sessions/x", None)]:
        kw = {"json": body} if body is not None else {}
        assert getattr(c, method)(path, **kw).status_code == 401
        assert getattr(c, method)(path, headers={"X-API-Key": "wrong"}, **kw).status_code == 401
    ok = c.post("/chat", json={"message": "q"}, headers={"X-API-Key": "s3cret"})
    assert ok.status_code == 200 and ok.json()["status"] == "ok"


def test_api_is_open_when_no_key_configured(monkeypatch):
    monkeypatch.delenv("PQ_API_KEY", raising=False)
    c, _ = client()
    assert c.post("/chat", json={"message": "q"}).status_code == 200
