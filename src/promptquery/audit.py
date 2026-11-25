"""Append-only JSONL audit trail: one record per attempt (including each repair round)."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class AuditLog:
    """Thread-safe JSONL writer. ``path=None`` disables logging (used in tests)."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: str, **fields: Any) -> dict[str, Any]:
        entry = {"ts": round(time.time(), 3), "event": event, **fields}
        if self.path:
            line = json.dumps(entry, default=str, ensure_ascii=False)
            with self._lock, self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        return entry
