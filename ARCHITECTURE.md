# PromptQuery — Architecture

This document explains how PromptQuery is structured internally. It's written for contributors and the curious. For usage, see the [README](README.md). This file covers two pipelines: the original single-turn CLI (below) and the multi-turn **Atlas report bot** (see [Atlas report bot pipeline](#atlas-report-bot-pipeline); full detail in [docs/ATLAS.md](docs/ATLAS.md)).

## The request lifecycle

When you type a question into the REPL, here is what happens:

```
question
   │
   ▼
┌──────────────────┐
│ retrieval.py     │  Rank tables by TF-IDF over their name, comment, column
│ TfIdfRetriever   │  names, and FK targets. Question is tokenized with
│                  │  snake_case + camelCase splitting and stopword filtering.
└────────┬─────────┘  Returns top-K tables.
         │
         ▼
┌──────────────────┐
│ retrieval.py     │  Walk the FK graph one hop outward (referenced tables)
│ expand_via_fks   │  AND one hop inward (tables that reference the seed).
│                  │  Capped at --max-tables to keep prompts compact.
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ prompts.py       │  Render the chosen tables into the system prompt:
│ format_schema    │  TABLE name, columns with PK/NOT NULL flags, column
│                  │  comments, the legal values of every enum column, FKs.
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ llm.py           │  Send to the configured provider (Anthropic by default,
│ Client.generate  │  OpenAI as fallback). Response is a markdown ```sql block.
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ llm.extract_sql  │  Pull the SQL out of the code fence.
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ safety.py        │  Parse with sqlglot in Postgres dialect. Reject anything
│ validate_select  │  that is not a single SELECT / WITH / UNION /
│ _only            │  INTERSECT / EXCEPT. Reject CTEs that hide DML
│                  │  (WITH x AS (DELETE ... RETURNING ...) SELECT ...).
│                  │  Reject calls to dangerous functions
│                  │  (pg_terminate_backend, set_config, lo_export, ...).
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ render.py        │  Pretty-print the SQL with syntax highlighting.
│ render_sql       │
└────────┬─────────┘
         │
         ▼
   "Run? [y/N]"
         │
         ▼
┌──────────────────┐
│ db.py            │  Execute. The Postgres session was opened with
│ Database.execute │  default_transaction_read_only = on and a 60s
│                  │  statement_timeout, so even if safety.py failed
│                  │  the database itself would refuse a write.
└────────┬─────────┘
         │ on a database error: repair.py feeds the failed SQL plus the
         │ database's own error message back to the model for up to
         │ --max-repair rounds (default 1). Every repaired query goes
         │ through validate_select_only — and the confirm prompt, in the
         │ REPL — before it is executed. Empty results never trigger a
         │ repair: an empty result is often the right answer.
         ▼
┌──────────────────┐
│ render.py        │  Format as a rich.Table. NULLs styled, large blobs
│ render_results   │  summarized, truncate after 100 rows by default.
└──────────────────┘
```

## File inventory (`src/promptquery/`)

| File              | Purpose                                                                   |
|-------------------|---------------------------------------------------------------------------|
| `__init__.py`     | Package version.                                                          |
| `__main__.py`     | `python -m promptquery` entry point.                                      |
| `cli.py`          | Click command and prompt-toolkit REPL. Orchestrates the whole pipeline.   |
| `db.py`           | psycopg3 connection wrapper. Sets the read-only session.                  |
| `schema.py`       | Dataclasses + the `pg_catalog` queries that introspect them.              |
| `retrieval.py`    | Tokenizer, TF-IDF ranker, FK-graph expander.                              |
| `llm.py`          | Provider clients (Anthropic, OpenAI), SQL extractor, provider factory.    |
| `prompts.py`      | System prompt template and schema-to-prompt formatter.                    |
| `repair.py`       | Execution-guided repair: bounded retry loop fed by the database's errors. |
| `safety.py`       | The sqlglot-based query guard.                                            |
| `render.py`       | SQL syntax rendering and result-table rendering with rich.                |
| `engine.py`       | Atlas bot: `Engine.ask(session_id, text)`, the single multi-turn entry point. |
| `bootstrap.py`    | Atlas bot: environment settings -> a ready `Engine`.                      |
| `atlas_cli.py`    | Atlas bot CLI (`prq-atlas ask` / `chat`).                                 |
| `audit.py`        | Atlas bot: append-only JSONL audit trail.                                 |
| `api/app.py`      | Atlas bot: FastAPI service (`/chat`, `/sessions`), optional API key.      |
| `conversation/`   | Atlas bot: session state, follow-up rewriting, untrusted-data fencing.    |
| `atlas/`          | Atlas bot: curated schema pack, examples, as-of date, role/RLS SQL generator. |

## File relationships

- **`schema.py` ↔ `retrieval.py`** — the ranker reads `Table.name`, `Table.comment`, `Table.columns`, and `Table.foreign_keys`. Adding a field to `Table` should prompt a decision about weighting it in `_table_terms`.
- **`schema.py` ↔ `prompts.py`** — `format_schema` walks the same dataclasses. Schema additions usually need a prompt update.
- **`cli.py` ↔ everything else** — the only file that knows the full pipeline. New stages (query history, post-execution feedback) wire in here.
- **`safety.py` ↔ `llm.py`** — `extract_sql` runs first; `validate_select_only` runs second. Together they handle the case where the model returns malformed output.
- **`db.py` ↔ `safety.py`** — two layers, intentionally redundant. Either alone is insufficient; both together make a write impossible.
- **`repair.py` ↔ `safety.py`** — every repaired query is re-validated before execution. The repair loop widens what the model can fix; it never widens what can run.

## Design bets

### Why TF-IDF, not embeddings (yet)

TF-IDF works the moment you connect to a database. No model to download, no GPU, no API call to compute embeddings for hundreds of tables. The cost is that it cannot reason about synonyms — "customer" and "user" are different tokens to TF-IDF. Since 0.2 the LLM table-selector covers that gap (it sees the TF-IDF candidates and picks semantically), and measured retrieval recall on a 211-table benchmark sits at 98–100% without embeddings. Embeddings stay off the default path until the data shows a gap they would close.

### Why a separate FK-expansion pass

Many natural-language questions reference one core entity but require joins through several others. Asking "what's the average order value per country" needs `orders` (the main entity), but also `users` and `countries` (join targets). TF-IDF alone is unlikely to rank `countries` highly because the word "country" only appears once in its schema. The FK-graph walk catches join targets that pure relevance would miss.

### Why two safety layers

`safety.py` is the primary guard. It parses every statement and rejects anything other than a SELECT. The Postgres session-level `default_transaction_read_only = on` is the fallback: if a malicious prompt somehow produces SQL that the validator misclassifies (a parser bug, an unknown construct), Postgres itself refuses the write.

This redundancy is not paranoia. AI-generated SQL is, by construction, less predictable than human-written SQL. The cost of one accidental `DELETE` is high enough that doubling up is the only sensible default.

### The Atlas path adds more layers

The Atlas bot does not rely on those two layers alone: a restricted database role with an
allowlist of tables and columns, Row-Level Security keyed on the tenant, SQL-level tenant scoping,
an allowlist validator (`safety.validate_report_query`), per-call read-only transactions with row
caps and timeouts. See [docs/ATLAS.md](docs/ATLAS.md). The original `promptquery` CLI is unchanged
in behaviour apart from a stricter `validate_select_only` (it now also rejects `SELECT ... INTO`,
`FOR UPDATE`, `SET`/transaction control and more server-admin functions).

### Why introspect via `pg_catalog` instead of `information_schema`

`pg_catalog` is faster, more complete, and lets us read table comments via `obj_description`. We use `LATERAL unnest WITH ORDINALITY` to join `pg_constraint.conkey` with `pg_constraint.confkey` by ordinal position — the only correct way to handle composite foreign keys. For multi-tenant schemas such as Atlas, where every FK is `(tenant_id, x_id)`, the `tenant_id` half of a composite FK is dropped: otherwise every table would link to every other.

## What's intentionally not here

These are out of scope for now. They are tracked in the [roadmap](README.md#roadmap).

- MySQL / SQLite support — needs an adapter abstraction first.
- Multi-database sessions in one REPL.
- Data visualization (charts, plots).
- Query-history persistence between sessions in the original CLI (the Atlas bot keeps in-memory multi-turn sessions).
- Embedding-based retrieval.
- MCP server mode.

## Testing

```bash
pytest                      # offline: no database or API key needed
PQ_TEST_OWNER_DSN=... PQ_TEST_BOT_DSN=... pytest    # also runs the live Atlas tests
```

Offline tests use in-memory fakes for the repair loop, prompt serialization, safety guards,
retrieval, the engine (scripted model and fake database), the API and the evaluation comparator.
With a seeded Atlas database (see [docs/ATLAS.md](docs/ATLAS.md)) the live tests also check the
role/RLS boundary, tenant isolation, schema-pack drift, every metric, example and scenario query,
and full-stack conversations.

The most safety-critical files are `tests/test_safety.py` and `tests/test_safety_report.py`. Cases
there encode "things the validator MUST reject." Add cases when you discover new attack vectors; do
not delete cases during refactors.

## Atlas report bot pipeline

```
question -> conversation/rewrite.py   classify follow-up, resolve "the first one" from real rows
         -> atlas/examples.py          retrieve similar verified Q->SQL examples
         -> engine.py + atlas/prompts  generate <plan> + SQL from the curated pack
         -> atlas/asof.py              pin CURRENT_DATE/now() to the as-of date
         -> safety.validate_report_query   allowlists, star expansion, tenant scoping, LIMIT
         -> db.run_scoped              BEGIN READ ONLY, tenant set, timeout, row cap
         -> repair loop                validator error / DB error / suspicious empty result
         -> summary + audit.py
```

## Project layout

```
promptquery/
├── ARCHITECTURE.md          this file
├── docs/ATLAS.md            Atlas report bot guide (setup, safety model, evaluation)
├── pyproject.toml           hatchling; scripts: promptquery, prq, pquery, prq-atlas
├── Dockerfile, .env.example report-bot service packaging and configuration
├── sql/atlas_bot_role.sql   generated role + grants + RLS for Atlas
├── src/promptquery/         the package (see the file inventory above)
│   ├── atlas/pack/          curated schema pack (tables, enums, joins, metrics, glossary, examples)
│   ├── conversation/        multi-turn state and follow-up handling
│   └── api/                 FastAPI service
├── eval/                    original benchmarks; eval/atlas/ is the multi-turn Atlas harness
└── tests/                   pytest suite (offline + optional live Atlas tests)
```

## Contributing

PRs welcome. Run `pytest` before opening one. Keep the safety test suite untouched unless you're adding cases.
