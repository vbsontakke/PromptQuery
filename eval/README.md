# PromptQuery eval suite

Internal QA harness for the schema-retrieval pipeline. Runs on every commit, catches regressions before they ship.

## What it measures

For each natural-language question in the suite:

1. Run `TfIdfRetriever.rank(question, top_k=K)` against the test schema.
2. Walk the FK graph one hop in each direction (`expand_via_fks`).
3. Check whether the **must-retrieve** tables (hand-labeled per question) are present in the output.

A question **passes** if every must-retrieve table is in the final post-FK-expansion set. Partial coverage still contributes to recall@K metrics.

## Why this shape

Retrieval is the headline differentiator. If the retriever misses the relevant tables, no amount of LLM cleverness will produce correct SQL — so retrieval quality dominates end-to-end accuracy. Isolating it from the LLM also removes nondeterminism: the eval is **pure-Python, free, and runs in <100 ms for the full suite**.

End-to-end evaluation (LLM generation + execution comparison against reference SQL) lives in separate runners because it needs a live Postgres and an API key, and the signal carries LLM variance: `eval/end_to_end.py` (shop schema, result-set equality), `eval/parsing_bench.py` (table-reference check on large schemas) and `eval/atlas/` (multi-turn Atlas scenarios, below).

## Run it

From the repo root:

```bash
.venv/bin/python -m eval.retrieval
```

Options:

| Flag | Default | Meaning |
|---|---|---|
| `--top-k` | 10 | Top-K used by the TF-IDF retriever |
| `--max-tables` | 20 | Cap after FK expansion |
| `--quiet` | off | Only print the summary table |
| `--fail-under <f>` | none | Exit non-zero if pass rate < f (for CI gating) |
| `--fixture <path>` | `fixtures/shop.schema.json` | Alternate schema |

Wire it into CI as a regression gate:

```bash
python -m eval.retrieval --quiet --fail-under 0.85
```

## What's in the suite

- **Schema**: hand-crafted 14-table e-commerce shop (`fixtures/shop.sql` is the DDL; `fixtures/shop.schema.json` is the pre-introspected fixture the eval consumes).
- **Questions**: 25 hand-written, in `questions/shop.py`, spanning easy → hard difficulty, single-table → 4-table joins, including window functions and time-range buckets.

## Adding a question

Open `questions/shop.py` and append a `Question(...)`. Required fields:

- `text` — what the user types
- `must_retrieve` — tuple of tables the retriever MUST surface (the contract)
- `reference_sql` — a correct answer (used by the future end-to-end eval; not scored here)
- `difficulty` — `"easy"` / `"medium"` / `"hard"`
- `tags` — free-form for failure-mode analysis

If your question targets a domain not yet in the schema, add the tables to `fixtures/shop.sql` AND `fixtures/shop.schema.json` together, then re-run the eval.

## Adding a schema fixture

To benchmark a different schema:

1. Build the DDL → save to `fixtures/<name>.sql`.
2. Either spin up a Postgres + run `promptquery.schema.introspect()` and dump `to_dict()` as JSON, or hand-craft the JSON. Save to `fixtures/<name>.schema.json`.
3. Author a `questions/<name>.py` exporting a `QUESTIONS` list.
4. Add a CLI flag or a new module under `eval/` for it.

## Atlas multi-turn evaluation (`eval/atlas/`)

92 scenarios / 197 turns (16 six-turn conversations) against a seeded Atlas ERP database, each turn
with a gold SQL scored by execution accuracy through the same tenant-scoped, read-only path the
bot uses. Metrics: end-to-end and teacher-forced per-turn accuracy, per-scenario accuracy, by
module/turn index, refusal accuracy, repair rate. Run, tune on `dev`, and use `heldout` sparingly:

```bash
python -m eval.atlas.run --split dev --mode both --cache .cache/llm --out report.json
python -m eval.atlas.run --split all --oracle      # harness self-check, no API key
python -m eval.atlas.run --split dev --fail-under 0.95
```

Full description, scenario format and setup: [docs/ATLAS.md](../docs/ATLAS.md).

## Roadmap

- **End-to-end suite** — implemented (`end_to_end.py`, `atlas/`); results from older runs under `eval/results*.json` predate the v0.2 pipeline and are kept as receipts only.
- **Scaled retrieval suite** — inject N=100/500/1000 noise tables into the fixture to test retrieval-at-scale (the headline pitch). Pass criteria becomes "must-retrieve still in top 20 with 1000-table haystack."
- **Cross-tool comparison** — adapters that run Vanna, WrenAI, LangChain SQLDatabaseChain, and LlamaIndex against the same suite. The public benchmark post writes itself.
