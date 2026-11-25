"""Draft the enum part of the Atlas pack from the Atlas backend source.

Atlas stores every enum as a plain VARCHAR, so ``pg_catalog`` cannot tell the generator which
values are legal ("is the open state ``OPEN`` or ``POSTED``?"). The truth lives in two places
in the Atlas source:

* ``StrEnum`` classes in ``modules/*/constants*`` (values + a docstring describing each state);
* SQLAlchemy models, where a column is tied to its enum via ``default=SomeEnum.MEMBER.value``.

This parses both with ``ast`` (no Atlas imports, so it needs none of Atlas's dependencies) and
emits ``enums.yaml``: ``{table: {column: {enum, values, doc}}}``. A human then reviews it and
adds the columns the heuristic cannot see (see ``OVERRIDES`` in ``enums_overrides.yaml``).
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class EnumDef:
    name: str
    values: list[str]
    doc: str


def _first_paragraph(doc: str | None) -> str:
    if not doc:
        return ""
    para = doc.strip().split("\n\n")[0]
    return " ".join(line.strip() for line in para.splitlines()).strip()


def parse_enums(path: Path) -> dict[str, EnumDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: dict[str, EnumDef] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        bases = {b.id for b in node.bases if isinstance(b, ast.Name)}
        if "StrEnum" not in bases and "Enum" not in bases:
            continue
        values: list[str] = []
        for stmt in node.body:
            if (
                isinstance(stmt, ast.Assign)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            ):
                values.append(stmt.value.value)
        if values:
            found[node.name] = EnumDef(node.name, values, _first_paragraph(ast.get_docstring(node)))
    return found


def _enum_ref(call: ast.Call) -> str | None:
    """Return ``Enum`` from ``default=Enum.MEMBER.value`` / ``default=Enum.MEMBER``."""
    for kw in call.keywords:
        if kw.arg != "default":
            continue
        node = kw.value
        if isinstance(node, ast.Attribute) and node.attr == "value":
            node = node.value
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            return node.value.id
    return None


def parse_model_enum_columns(path: Path) -> dict[str, dict[str, str]]:
    """``{table: {column: EnumName}}`` for every column whose default names an enum member."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, str]] = {}
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        table = None
        for stmt in cls.body:
            if (
                isinstance(stmt, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "__tablename__" for t in stmt.targets)
                and isinstance(stmt.value, ast.Constant)
            ):
                table = stmt.value.value
        if not table:
            continue
        for stmt in cls.body:
            if not (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)):
                continue
            call = stmt.value
            if isinstance(call, ast.Call) and getattr(call.func, "id", "") == "mapped_column":
                enum = _enum_ref(call)
                if enum:
                    out.setdefault(table, {})[stmt.target.id] = enum
    return out


def build_enum_map(
    atlas_root: Path, overrides: dict[str, dict[str, object]] | None = None
) -> dict[str, dict[str, dict]]:
    """Map ``table -> column -> {enum, values, doc}``.

    ``overrides`` is ``{table: {column: EnumClassName | [explicit, values]}}`` for the columns
    whose model declares no enum default (so the AST pass cannot see the link). An override
    that names an unknown enum raises, so a renamed Atlas enum cannot silently drop values.
    """
    modules = atlas_root / "backend" / "app" / "modules"
    enums: dict[str, EnumDef] = {}
    for p in list(modules.rglob("constants.py")) + list(modules.rglob("constants/*.py")):
        enums.update(parse_enums(p))
    result: dict[str, dict[str, dict]] = {}
    model_files = list(modules.glob("*/models.py")) + list(modules.glob("*/models/*.py"))
    for p in model_files:
        for table, cols in parse_model_enum_columns(p).items():
            for col, enum_name in cols.items():
                enum = enums.get(enum_name)
                if enum is None:
                    continue
                result.setdefault(table, {})[col] = {
                    "enum": enum.name,
                    "values": enum.values,
                    "doc": enum.doc,
                }
    for table, cols in (overrides or {}).items():
        for col, spec in cols.items():
            if isinstance(spec, list):
                result.setdefault(table, {})[col] = {"enum": None, "values": list(spec), "doc": ""}
                continue
            enum = enums.get(str(spec))
            if enum is None:
                raise KeyError(f"override {table}.{col}: unknown enum {spec!r}")
            result.setdefault(table, {})[col] = {
                "enum": enum.name,
                "values": enum.values,
                "doc": enum.doc,
            }
    return {t: result[t] for t in sorted(result)}


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - thin CLI
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        raise SystemExit("usage: gen_pack_skeleton ATLAS_ROOT OUT_YAML")
    overrides_path = Path(__file__).parent / "pack" / "enums_overrides.yaml"
    overrides = yaml.safe_load(overrides_path.read_text(encoding="utf-8")) or {}
    mapping = build_enum_map(Path(argv[0]), overrides)
    Path(argv[1]).write_text(
        yaml.safe_dump(mapping, sort_keys=False, allow_unicode=True, width=100), encoding="utf-8"
    )
    cols = sum(len(v) for v in mapping.values())
    print(f"wrote {cols} enum columns across {len(mapping)} tables -> {argv[1]}")


if __name__ == "__main__":  # pragma: no cover
    main()
