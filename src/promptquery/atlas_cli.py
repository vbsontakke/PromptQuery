"""Command line front-end for the Atlas report bot (thin wrapper over ``Engine``).

    prq-atlas ask "total open receivables"            one question
    prq-atlas chat                                    interactive, up to 6 questions
    prq-atlas chat --script prompts.txt --json        run questions from a file, in order

Configuration comes from the environment; see ``promptquery.bootstrap``.
"""

from __future__ import annotations

import json
import sys
import uuid

import click
from rich.console import Console
from rich.table import Table

from .bootstrap import ConfigError, build_engine, settings_from_env
from .conversation.context import sanitize_cell
from .engine import Engine, TurnResult
from .llm import LLMError

console = Console()


def _render(result: TurnResult, as_json: bool) -> None:
    if as_json:
        click.echo(json.dumps(result.to_dict(), ensure_ascii=False))
        return
    if result.status == "ok":
        if result.sql:
            console.print(f"[dim]{result.sql}[/dim]")
        table = Table(show_lines=False)
        for c in result.columns:
            table.add_column(str(c))
        for row in result.rows[:50]:
            table.add_row(*[sanitize_cell(v) for v in row])
        console.print(table)
        shown = min(len(result.rows), 50)
        extra = " (truncated)" if result.truncated else ""
        console.print(f"[dim]{result.row_count} row(s){extra}; showing {shown}[/dim]")
    elif result.status == "turn_limit":
        console.print(f"[yellow]{result.answer_text}[/yellow]")
    else:
        console.print(f"[red]{result.status}:[/red] {result.answer_text or result.error}")


def _engine() -> Engine:
    try:
        return build_engine(settings_from_env())
    except (ConfigError, LLMError) as e:
        raise click.ClickException(str(e)) from e


@click.group()
def main() -> None:
    """Ask business questions about an Atlas ERP database in plain English."""


@main.command()
@click.argument("question")
@click.option("--json", "as_json", is_flag=True, help="Print the result as one JSON object.")
def ask(question: str, as_json: bool) -> None:
    """Answer a single QUESTION."""
    result = _engine().ask(uuid.uuid4().hex, question)
    _render(result, as_json)
    sys.exit(0 if result.status in ("ok", "refused") else 1)


@main.command()
@click.option("--script", type=click.Path(exists=True, dir_okay=False),
              help="File with one question per line (blank lines and # comments skipped).")
@click.option("--json", "as_json", is_flag=True, help="Print each result as one JSON object.")
def chat(script: str | None, as_json: bool) -> None:
    """Start a multi-turn conversation (at most 6 questions per session)."""
    engine = _engine()
    sid = uuid.uuid4().hex
    if script:
        with open(script, encoding="utf-8") as fh:
            for line in fh:
                q = line.strip()
                if q and not q.startswith("#"):
                    _render(engine.ask(sid, q), as_json)
        return
    console.print("Ask a question. Commands: /reset  /quit   (max 6 questions per conversation)")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q in ("/quit", "/exit"):
            break
        if q == "/reset":
            engine.reset_session(sid)
            sid = uuid.uuid4().hex
            console.print("[dim]new conversation[/dim]")
            continue
        _render(engine.ask(sid, q), as_json)


if __name__ == "__main__":  # pragma: no cover
    main()
