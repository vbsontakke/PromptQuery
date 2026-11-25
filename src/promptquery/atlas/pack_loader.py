"""Merge live schema introspection with the curated Atlas pack (descriptions, enums, joins).

Introspection alone cannot tell the model what Atlas needs it to know: enums are plain VARCHAR,
97 cross-module ``*_id`` columns have no foreign key, and "posted"/"open" are business rules.
The pack supplies that. Everything it renders is derived from the live schema first, so a column
that does not exist in the database can never reach the prompt.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..schema import Schema, Table
from .security import allowed_columns, allowed_tables

PACK_DIR = Path(__file__).parent / "pack"

# Columns that exist on (almost) every table and only add noise to the prompt. They remain
# queryable (RLS needs tenant_id) but are described once in the preamble.
_HIDDEN_IN_PROMPT = frozenset({"tenant_id", "created_at", "updated_at", "document_id"})

_MODULE_BY_PREFIX = {
    "fin_": "finance", "sales_": "sales", "proc_": "procurement", "inv_": "inventory",
    "mfg_": "manufacturing", "hr_": "hr", "crm_": "crm", "ps_": "projects", "qm_": "quality",
    "pm_": "maintenance", "hsp_": "hospitality", "core_": "core", "adm_": "core", "ind_": "core",
}


class PackDriftError(Exception):
    """The pack no longer matches the database (a table/column was added, renamed or removed)."""


@dataclass
class PackColumn:
    name: str
    data_type: str
    nullable: bool
    is_pk: bool
    enum_values: tuple[str, ...] | None = None
    enum_name: str | None = None
    note: str = ""
    ref: str | None = None  # "table.column" this column points at (declared or inferred)
    ref_inferred: bool = False
    ref_when: str | None = None


@dataclass
class PackTable:
    name: str
    module: str
    desc: str
    internal: bool
    columns: list[PackColumn] = field(default_factory=list)


@dataclass
class Metric:
    key: str
    synonyms: list[str]
    definition: str
    sql: str


@dataclass
class Pack:
    tables: dict[str, PackTable]
    metrics: dict[str, Metric]
    glossary: dict[str, str]
    recipes: list[str]
    as_of: dt.date
    readable_columns: dict[str, list[str]]
    explicit_joins: list[dict[str, Any]]

    # -- validator / security ----------------------------------------------------------
    def allowed_columns(self) -> dict[str, list[str]]:
        return self.readable_columns

    @property
    def modules(self) -> list[str]:
        return sorted({t.module for t in self.tables.values() if not t.internal})

    # -- rendering ---------------------------------------------------------------------
    def render_metrics(self, keys: list[str] | None = None) -> str:
        out: list[str] = []
        for key, m in self.metrics.items():
            if keys is not None and key not in keys:
                continue
            sql = m.sql.replace("{as_of}", self.as_of.isoformat()).strip()
            out.append(f"### {key} - {m.definition.strip()}\n```sql\n{sql}\n```")
        return "\n".join(out)

    def render_schema(self, modules: list[str] | None = None, stubs: bool = True) -> str:
        """Compact schema text. Tables of ``modules`` are rendered in full, the rest as
        one-line stubs (so cross-module questions can still name them)."""
        blocks: list[str] = []
        for module in sorted({t.module for t in self.tables.values() if not t.internal}):
            full = modules is None or module in modules
            lines = [f"## {module}"]
            for t in sorted(self.tables.values(), key=lambda x: x.name):
                if t.module != module or t.internal:
                    continue
                if full:
                    lines.append(_render_table(t))
                elif stubs:
                    lines.append(f"{t.name} - {t.desc}")
            if len(lines) > 1:
                blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def render_joins(self) -> str:
        lines = []
        for j in self.explicit_joins:
            extra = f" when {j['when']}" if j.get("when") else ""
            note = f" ({j['note']})" if j.get("note") else ""
            lines.append(f"- {j['from']} = {j['to']}{extra}{note}")
        lines.extend(f"- {r}" for r in self.recipes)
        return "\n".join(lines)

    def render_glossary(self) -> str:
        return "\n".join(f"- {k}: {v}" for k, v in self.glossary.items())


def _short_type(t: str) -> str:
    t = t.lower()
    if t.startswith(("character varying", "varchar", "text", "character")):
        return "text"
    if t.startswith("numeric"):
        return "numeric"
    if t.startswith("timestamp with"):
        return "timestamptz"
    if t.startswith("timestamp"):
        return "timestamp"
    if t == "boolean":
        return "bool"
    return t


def _render_table(t: PackTable) -> str:
    lines = [f"{t.name} - {t.desc}"]
    for c in t.columns:
        if c.name in _HIDDEN_IN_PROMPT:
            continue
        bits = [c.name, _short_type(c.data_type)]
        if c.is_pk:
            bits.append("PK")
        if c.ref:
            when = f" when {c.ref_when}" if c.ref_when else ""
            bits.append(f"->{c.ref}{when}")
        if c.enum_values:
            bits.append("{" + "|".join(c.enum_values) + "}")
        line = "  " + " ".join(bits)
        if c.note:
            line += f"  -- {c.note}"
        lines.append(line)
    return "\n".join(lines)


def _module_of(name: str) -> str:
    for prefix, module in _MODULE_BY_PREFIX.items():
        if name.startswith(prefix):
            return module
    return "other"


def _load_yaml(name: str, pack_dir: Path) -> Any:
    return yaml.safe_load((pack_dir / name).read_text(encoding="utf-8")) or {}


def load_pack(
    schema: Schema,
    as_of: dt.date,
    *,
    pack_dir: Path | None = None,
    strict: bool = False,
) -> Pack:
    pack_dir = pack_dir or PACK_DIR
    tables_yaml = _load_yaml("tables.yaml", pack_dir).get("tables", {})
    enums = _load_yaml("enums.yaml", pack_dir)
    joins_yaml = _load_yaml("joins.yaml", pack_dir)
    metrics_yaml = _load_yaml("metrics.yaml", pack_dir).get("metrics", {})
    glossary = _load_yaml("glossary.yaml", pack_dir).get("terms", {})

    readable = allowed_tables(schema)
    by_name: dict[str, Table] = {t.name: t for t in schema.tables if t.schema == "public"}
    rules: dict[str, str] = joins_yaml.get("rules", {})
    explicit = joins_yaml.get("explicit", [])
    explicit_by_col: dict[str, dict] = {e["from"]: e for e in explicit}

    problems = check_pack(schema, tables_yaml, enums, joins_yaml)
    if strict and problems:
        raise PackDriftError("; ".join(problems))

    tables: dict[str, PackTable] = {}
    for name in readable:
        meta = tables_yaml.get(name, {})
        db_table = by_name[name]
        readable_cols = set(allowed_columns(schema, name))
        fk_by_col = {fk.column: fk for fk in db_table.foreign_keys}
        notes = meta.get("cols", {})
        enum_cols = enums.get(name, {})
        cols: list[PackColumn] = []
        for c in db_table.columns:
            if c.name not in readable_cols:
                continue
            col = PackColumn(
                name=c.name, data_type=c.data_type, nullable=c.nullable, is_pk=c.is_primary_key,
                note=notes.get(c.name, ""),
            )
            e = enum_cols.get(c.name)
            if e:
                col.enum_values = tuple(e["values"])
                col.enum_name = e.get("enum")
            elif c.enum_values:
                col.enum_values = tuple(c.enum_values)
            fk = fk_by_col.get(c.name)
            key = f"{name}.{c.name}"
            if fk is not None and fk.referenced_table in by_name:
                col.ref = f"{fk.referenced_table}.{fk.referenced_column}"
            elif key in explicit_by_col and not explicit_by_col[key].get("when"):
                col.ref = explicit_by_col[key]["to"]
                col.ref_inferred = True
            elif c.name in rules and rules[c.name] in by_name and c.name != "id":
                col.ref = f"{rules[c.name]}.id"
                col.ref_inferred = True
            cols.append(col)
        # polymorphic explicit joins (with `when`) are attached to the column's first entry
        for c in cols:
            conditional = [
                e for e in explicit if e["from"] == f"{name}.{c.name}" and e.get("when")
            ]
            if conditional and c.ref is None:
                c.ref = " | ".join(f"{e['to']} [{e['when']}]" for e in conditional)
        tables[name] = PackTable(
            name=name,
            module=meta.get("module") or _module_of(name),
            desc=meta.get("desc", ""),
            internal=bool(meta.get("internal", False)),
            columns=cols,
        )

    metrics = {
        k: Metric(k, list(v.get("synonyms", [])), v.get("definition", ""), v.get("sql", ""))
        for k, v in metrics_yaml.items()
    }
    return Pack(
        tables=tables,
        metrics=metrics,
        glossary=glossary,
        recipes=list(joins_yaml.get("recipes", [])),
        as_of=as_of,
        readable_columns={n: allowed_columns(schema, n) for n in readable},
        explicit_joins=list(explicit),
    )


def check_pack(
    schema: Schema, tables_yaml: dict, enums: dict, joins_yaml: dict
) -> list[str]:
    """Return human-readable drift problems between the pack files and the live schema."""
    problems: list[str] = []
    by_name = {t.name: t for t in schema.tables if t.schema == "public"}
    readable = set(allowed_tables(schema))

    for name in sorted(readable - set(tables_yaml)):
        problems.append(f"table {name!r} is in the database but not in tables.yaml")
    for name in sorted(set(tables_yaml) - set(by_name)):
        problems.append(f"tables.yaml lists {name!r} which is not in the database")

    for name, meta in tables_yaml.items():
        t = by_name.get(name)
        if t is None:
            continue
        have = {c.name for c in t.columns}
        for col in meta.get("cols", {}):
            if col not in have:
                problems.append(f"tables.yaml {name}.{col}: no such column")
    for name, cols in enums.items():
        t = by_name.get(name)
        if t is None:
            problems.append(f"enums.yaml lists table {name!r} which is not in the database")
            continue
        have = {c.name for c in t.columns}
        for col in cols:
            if col not in have:
                problems.append(f"enums.yaml {name}.{col}: no such column")

    for col, target in joins_yaml.get("rules", {}).items():
        if target not in by_name:
            problems.append(f"joins.yaml rule {col} -> {target!r}: no such table")
    for e in joins_yaml.get("explicit", []):
        for side in ("from", "to"):
            m = re.fullmatch(r"(\w+)\.(\w+)", e[side])
            if not m or m.group(1) not in by_name:
                problems.append(f"joins.yaml explicit {e[side]!r}: unknown table")
            elif m.group(2) not in {c.name for c in by_name[m.group(1)].columns}:
                problems.append(f"joins.yaml explicit {e[side]!r}: unknown column")
    return problems
