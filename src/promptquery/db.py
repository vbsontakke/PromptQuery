from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import psycopg
from psycopg.rows import dict_row

DEFAULT_ROW_CAP = 1000
DEFAULT_TIMEOUT_MS = 15_000
TENANT_SETTING = "app.tenant_id"


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[tuple] = field(default_factory=list)
    # True when more rows existed than the cap; ``rows`` holds exactly the cap.
    truncated: bool = False


class Database:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.conn: psycopg.Connection | None = None

    def connect(self) -> None:
        self.conn = psycopg.connect(self.dsn, autocommit=True)
        with self.conn.cursor() as cur:
            cur.execute("SET default_transaction_read_only = on")
            cur.execute("SET statement_timeout = '60s'")

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def _require_conn(self) -> psycopg.Connection:
        if self.conn is None:
            raise RuntimeError("Database is not connected")
        return self.conn

    def fetch_dicts(self, sql: str) -> list[dict]:
        conn = self._require_conn()
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql)
            return list(cur.fetchall())

    def execute(self, sql: str) -> tuple[list[str], list[tuple]]:
        conn = self._require_conn()
        with conn.cursor() as cur:
            cur.execute(sql)
            if cur.description is None:
                return [], []
            cols = [d.name for d in cur.description]
            rows = list(cur.fetchall())
        return cols, rows

    def run_scoped(
        self,
        sql: str,
        *,
        tenant_id: str | uuid.UUID | None,
        max_rows: int = DEFAULT_ROW_CAP,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
    ) -> QueryResult:
        """Run one query in a read-only transaction scoped to a single tenant.

        - ``BEGIN READ ONLY`` is enforced per call, not merely as a session default.
        - ``app.tenant_id`` is set transaction-locally; Row-Level Security on the bot role
          keys on it, so the SQL never needs (and the model never controls) tenant filters.
        - At most ``max_rows`` rows are fetched (``truncated`` flags overflow), so a
          runaway ``SELECT *`` cannot exhaust memory.
        - The transaction is always rolled back.
        """
        conn = self._require_conn()
        if max_rows < 1:
            raise ValueError("max_rows must be >= 1")
        tenant = str(uuid.UUID(str(tenant_id))) if tenant_id is not None else None

        with conn.cursor() as cur:
            cur.execute("BEGIN READ ONLY")
            try:
                if tenant is not None:
                    cur.execute("SELECT set_config(%s, %s, true)", (TENANT_SETTING, tenant))
                cur.execute(
                    "SELECT set_config('statement_timeout', %s, true)", (f"{int(timeout_ms)}ms",)
                )
                cur.execute(sql)
                if cur.description is None:
                    return QueryResult([], [], False)
                cols = [d.name for d in cur.description]
                rows = cur.fetchmany(max_rows + 1)
            finally:
                try:
                    cur.execute("ROLLBACK")
                except psycopg.Error:
                    # The connection is unusable; surface it on the next call.
                    pass
        truncated = len(rows) > max_rows
        return QueryResult(cols, rows[:max_rows], truncated)

    def __enter__(self) -> "Database":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
