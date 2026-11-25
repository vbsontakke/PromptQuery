"""Conversation state: a session is an ordered list of at most MAX_TURNS turns."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

MAX_TURNS = 6
PREVIEW_ROWS = 20


@dataclass
class TurnRecord:
    turn_no: int
    user_text: str
    standalone_question: str
    intent: str  # NEW | REFINE | DRILL | REFERENCE | RESET | CLARIFY
    status: str  # ok | refused | clarify | error
    sql: str | None = None
    columns: list[str] = field(default_factory=list)
    preview_rows: list[tuple] = field(default_factory=list)  # first PREVIEW_ROWS rows
    row_count: int = 0
    truncated: bool = False
    answer_text: str = ""
    error: str | None = None


@dataclass
class Session:
    id: str
    tenant_id: str
    turns: list[TurnRecord] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def is_full(self) -> bool:
        return self.turn_count >= MAX_TURNS

    @property
    def last_ok(self) -> TurnRecord | None:
        return next((t for t in reversed(self.turns) if t.status == "ok" and t.sql), None)


class ConversationStore:
    """Thread-safe in-memory session store with a TTL and a size cap.

    Idle sessions expire after ``ttl_seconds`` and, past ``max_sessions``, the least recently
    used one is evicted, so a long-running service cannot grow without bound. One process only;
    swap in a shared store before running several workers.
    """

    def __init__(self, max_sessions: int = 1000, ttl_seconds: float = 3600.0,
                 clock=time.time) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self._clock = clock

    def _purge(self) -> None:
        now = self._clock()
        for sid in [k for k, v in self._sessions.items() if now - v.last_used > self.ttl_seconds]:
            del self._sessions[sid]
        while len(self._sessions) >= self.max_sessions:
            oldest = min(self._sessions, key=lambda k: self._sessions[k].last_used)
            del self._sessions[oldest]

    def create(self, tenant_id: str, session_id: str | None = None) -> Session:
        with self._lock:
            sid = session_id or uuid.uuid4().hex
            if sid in self._sessions:
                self._sessions[sid].last_used = self._clock()
                return self._sessions[sid]
            self._purge()
            s = Session(id=sid, tenant_id=tenant_id, created_at=self._clock(), last_used=self._clock())
            self._sessions[sid] = s
            return s

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            s = self._sessions.get(session_id)
            if s is None:
                return None
            if self._clock() - s.last_used > self.ttl_seconds:
                del self._sessions[session_id]
                return None
            s.last_used = self._clock()
            return s

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None
