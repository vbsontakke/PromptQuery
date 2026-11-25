"""Build a ready-to-use Engine from environment settings.

    PQ_DSN         connection string the bot queries with (ideally the restricted ``atlas_bot``
                   role from sql/atlas_bot_role.sql)                                [required]
    PQ_TENANT      tenant slug (needs PQ_ADMIN_DSN to resolve) or tenant UUID        [required]
    PQ_ADMIN_DSN   optional privileged DSN, used only to resolve a tenant slug to its id
    PQ_AS_OF       the conversation's "today", ISO date                       [default 2026-07-06]
    PQ_MODEL       SQL-generation model                                [default claude-sonnet-5-5]
    PQ_FAST_MODEL  follow-up rewriting model                            [default claude-haiku-5-5]
    PQ_AUDIT_LOG   path of the JSONL audit log                                       [optional]
    PQ_ROW_CAP     max rows returned per query                                [default 1000]
"""

from __future__ import annotations

import datetime as dt
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .atlas.examples import ExampleLibrary
from .atlas.pack_loader import load_pack
from .audit import AuditLog
from .db import Database
from .engine import Engine
from .llm import DEFAULT_FAST_MODEL, DEFAULT_MODEL, AnthropicClient, CachingLLMClient, LLMClient
from .schema import introspect

DEFAULT_AS_OF = "2026-07-06"  # the end of Atlas's seeded demo data window


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    dsn: str
    tenant: str
    admin_dsn: str | None = None
    as_of: dt.date = dt.date(2026, 7, 6)
    model: str = DEFAULT_MODEL
    fast_model: str = DEFAULT_FAST_MODEL
    audit_log: str | None = None
    row_cap: int = 1000
    llm_cache_dir: str | None = None


def settings_from_env(env: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    missing = [k for k in ("PQ_DSN", "PQ_TENANT") if not env.get(k)]
    if missing:
        raise ConfigError(f"Missing required environment variable(s): {', '.join(missing)}")
    try:
        as_of = dt.date.fromisoformat(env.get("PQ_AS_OF", DEFAULT_AS_OF))
    except ValueError as e:
        raise ConfigError(f"PQ_AS_OF must be an ISO date (YYYY-MM-DD): {e}") from e
    try:
        row_cap = int(env.get("PQ_ROW_CAP", "1000"))
    except ValueError as e:
        raise ConfigError("PQ_ROW_CAP must be an integer") from e
    if row_cap < 1:
        raise ConfigError("PQ_ROW_CAP must be >= 1")
    return Settings(
        dsn=env["PQ_DSN"], tenant=env["PQ_TENANT"], admin_dsn=env.get("PQ_ADMIN_DSN") or None,
        as_of=as_of, model=env.get("PQ_MODEL") or DEFAULT_MODEL,
        fast_model=env.get("PQ_FAST_MODEL") or DEFAULT_FAST_MODEL,
        audit_log=env.get("PQ_AUDIT_LOG") or None, row_cap=row_cap,
        llm_cache_dir=env.get("PQ_LLM_CACHE") or None,
    )


def resolve_tenant_id(tenant: str, admin_dsn: str | None, fallback_dsn: str) -> str:
    """Accept a UUID as-is; resolve a slug through ``adm_tenants`` with a privileged DSN."""
    try:
        return str(uuid.UUID(tenant))
    except ValueError:
        pass
    with Database(admin_dsn or fallback_dsn) as db:
        try:
            rows = db._require_conn().execute(
                "SELECT id FROM adm_tenants WHERE slug = %s", (tenant,)
            ).fetchall()
        except Exception as e:  # noqa: BLE001
            raise ConfigError(
                f"Cannot resolve tenant slug {tenant!r}: {e}. Use the tenant UUID in PQ_TENANT, "
                "or set PQ_ADMIN_DSN to a role that can read adm_tenants."
            ) from e
    if not rows:
        raise ConfigError(f"Unknown tenant slug {tenant!r}")
    return str(rows[0][0])


def build_engine(
    settings: Settings, *, llm: LLMClient | None = None, fast_llm: LLMClient | None = None
) -> Engine:
    tenant_id = resolve_tenant_id(settings.tenant, settings.admin_dsn, settings.dsn)
    db = Database(settings.dsn)
    db.connect()
    schema = introspect(db)
    pack = load_pack(schema, settings.as_of)
    if llm is None:
        llm = AnthropicClient(model=settings.model)
    if fast_llm is None:
        fast_llm = AnthropicClient(model=settings.fast_model)
    if settings.llm_cache_dir:
        llm = CachingLLMClient(llm, Path(settings.llm_cache_dir))
        fast_llm = CachingLLMClient(fast_llm, Path(settings.llm_cache_dir))
    return Engine(
        db=db, pack=pack, llm=llm, fast_llm=fast_llm, tenant_id=tenant_id,
        audit=AuditLog(settings.audit_log), row_cap=settings.row_cap,
        examples=ExampleLibrary.load(),
    )
