"""HTTP front-end for the report bot (FastAPI).

    POST   /sessions            -> {session_id}                  start a conversation
    POST   /chat                -> TurnResult JSON               ask the next question
    GET    /sessions/{id}       -> {session_id, turns, max_turns, history}
    DELETE /sessions/{id}       -> {deleted}                     reset a conversation
    GET    /healthz             -> {status}

``POST /chat`` creates the session on first use, so a client may simply pick a session_id (or
omit it for a one-off question). A conversation holds at most 6 questions; the 7th returns
``status: "turn_limit"``.
"""

from __future__ import annotations

import hmac
import os
import threading
import uuid
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from ..bootstrap import build_engine, settings_from_env
from ..conversation.state import MAX_TURNS
from ..engine import Engine

MAX_QUESTION_CHARS = 2000


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS)
    session_id: str | None = Field(default=None, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")


def create_app(engine: Engine | None = None, api_key: str | None = None) -> FastAPI:
    """``api_key`` (default: env ``PQ_API_KEY``) protects every route except /healthz: callers
    must send it as ``X-API-Key``. Unset means open (fine on localhost, not on a network)."""
    key = api_key if api_key is not None else os.environ.get("PQ_API_KEY")

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        if key and not (x_api_key and hmac.compare_digest(x_api_key, key)):
            raise HTTPException(status_code=401, detail="Missing or invalid API key")

    app = FastAPI(title="Atlas Report Bot", version="1.0.0")
    holder: dict[str, Engine | None] = {"engine": engine}
    lock = threading.Lock()  # one psycopg connection: serialise queries (simple and safe)

    def get_engine() -> Engine:
        if holder["engine"] is None:
            holder["engine"] = build_engine(settings_from_env())
        return holder["engine"]

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/sessions", dependencies=[Depends(require_key)])
    def create_session() -> dict[str, Any]:
        s = get_engine().new_session(uuid.uuid4().hex)
        return {"session_id": s.id, "max_turns": MAX_TURNS}

    @app.post("/chat", dependencies=[Depends(require_key)])
    def chat(req: ChatRequest) -> dict[str, Any]:
        eng = get_engine()
        sid = req.session_id or uuid.uuid4().hex
        with lock:
            result = eng.ask(sid, req.message)
        return result.to_dict()

    @app.get("/sessions/{session_id}", dependencies=[Depends(require_key)])
    def get_session(session_id: str) -> dict[str, Any]:
        s = get_engine().store.get(session_id)
        if s is None:
            raise HTTPException(status_code=404, detail="Unknown session")
        return {
            "session_id": s.id,
            "turns": s.turn_count,
            "max_turns": MAX_TURNS,
            "history": [
                {"turn_no": t.turn_no, "question": t.user_text, "status": t.status,
                 "sql": t.sql, "row_count": t.row_count}
                for t in s.turns
            ],
        }

    @app.delete("/sessions/{session_id}", dependencies=[Depends(require_key)])
    def delete_session(session_id: str) -> dict[str, bool]:
        return {"deleted": get_engine().store.delete(session_id)}

    return app


def app_factory() -> FastAPI:  # pragma: no cover - uvicorn --factory entry point
    return create_app()
