"""Everything the CLI draws in the terminal.

Learning notes
--------------
This module only *draws*. It never talks to the model, never reads input and
never exits; ``main.py`` decides *what happens*, this module decides *how it
looks*. Keeping the two apart has two benefits:

* ``main.py`` stays short and reads like a description of the program flow;
* every function here takes the ``Console`` to draw on as an argument, so tests
  can pass a console that writes into a string and check the exact output.

Rich basics used here:

* ``console.print("[bold red]text[/bold red]")``: square brackets are *markup*
  (styles). Text we did not write ourselves (model answers, tool output, error
  messages) must go through ``escape()``, or a "[0]" in it could be read as a
  style tag, or even break the output.
* ``Panel`` draws a box, ``Table`` a table, ``Markdown`` renders **bold**,
  lists, code blocks, ... in the terminal.
* ``console.is_terminal`` is False when output goes to a file or a pipe
  (``lightspeed ask "..." > answer.md``). Then we print plain text, because
  boxes and colours would end up in the file as junk characters.
"""

import json
from collections.abc import Iterable, Sequence

from pydantic import JsonValue
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.traceback import Traceback

from lightspeed_cli.client import ToolFinished, ToolStarted
from lightspeed_cli.schemas import ChatMessage, ChatResult, ToolDefinition

# Long tool output is shortened in --debug mode so it does not flood the screen.
MAX_DEBUG_OUTPUT = 300

CHAT_COMMANDS = {
    "/help": "Show this help",
    "/tools": "List the tools the assistant can use",
    "/history": "Show the questions asked so far",
    "/clear": "Forget the conversation and start a new one",
    "/exit": "Leave the chat (also: exit, quit, Ctrl+D)",
}


# --------------------------------------------------------------------------- #
# Answers
# --------------------------------------------------------------------------- #
def render_answer(console: Console, result: ChatResult, *, model: str) -> None:
    """Show the assistant's reply: in a panel on a terminal, plain otherwise."""
    if not console.is_terminal:
        # Raw text keeps the Markdown itself (**bold**, `code`) intact, which is
        # what you want in a saved file. soft_wrap: do not insert line breaks.
        console.print(result.reply, markup=False, highlight=False, soft_wrap=True)
        return
    console.print(
        Panel(
            Markdown(result.reply),
            title="[bold cyan]Lightspeed[/bold cyan]",
            title_align="left",
            subtitle=f"[dim]{escape(model)}[/dim]",
            subtitle_align="right",
            border_style="cyan",
        )
    )


def render_stats(console: Console, result: ChatResult) -> None:
    """One dim line of statistics, plus a warning if the answer was cut off."""
    if result.finish_reason == "length":
        console.print("[yellow]Note: the answer was cut off (length limit).[/yellow]")
    console.print(
        f"[dim]{result.iterations} request(s), {len(result.tool_runs)} tool "
        f"call(s), {result.usage.total_tokens} tokens[/dim]"
    )


# --------------------------------------------------------------------------- #
# Tool progress
# --------------------------------------------------------------------------- #
def render_tool_started(console: Console, event: ToolStarted) -> None:
    """Debug mode only: show the arguments the model chose."""
    console.print(
        f"[dim]  > {escape(event.name)} called with "
        f"{escape(_short_json(event.arguments))}[/dim]"
    )


def render_tool_finished(
    console: Console, event: ToolFinished, *, debug: bool = False
) -> None:
    """One line per finished tool; with ``debug``, also its (shortened) result."""
    if event.result.status == "success":
        outcome = "[green]ok[/green]"
    else:
        outcome = "[red]failed[/red]"
    console.print(
        f"[dim]  > {escape(event.name)}[/dim] {outcome} "
        f"[dim]({event.duration_seconds:.1f}s)[/dim]"
    )
    if debug:
        detail = (
            event.result.error
            if event.result.status == "error"
            else _short_json(event.result.output)
        )
        console.print(f"[dim]    {escape(detail or '')}[/dim]")


def render_debug_tools(console: Console, tools: list[dict[str, JsonValue]]) -> None:
    """Debug mode: show the exact tool schemas sent to the model."""
    console.print(f"[dim]Sending {len(tools)} tool schema(s) to the model:[/dim]")
    console.print_json(data=tools)


def _short_json(value: JsonValue) -> str:
    text = json.dumps(value, ensure_ascii=False)
    if len(text) <= MAX_DEBUG_OUTPUT:
        return text
    return text[:MAX_DEBUG_OUTPUT] + "..."


# --------------------------------------------------------------------------- #
# Tools table
# --------------------------------------------------------------------------- #
def render_tools_table(console: Console, definitions: Iterable[ToolDefinition]) -> None:
    table = Table(title="Available tools")
    table.add_column("Tool Name", style="bold cyan", no_wrap=True)
    table.add_column("Description")
    table.add_column("Parameters")
    for definition in definitions:
        table.add_row(
            definition.name,
            escape(definition.description),
            _describe_parameters(definition),
        )
    console.print(table)
    console.print("[dim]Bold parameters are required; others show their default.[/dim]")


def _describe_parameters(definition: ToolDefinition) -> str:
    """e.g. "[bold]path[/bold], [dim]max_commits=3[/dim]" (required in bold)."""
    schema = definition.parameters
    parts = []
    for name, prop in schema.properties.items():
        if name in schema.required:
            parts.append(f"[bold]{escape(name)}[/bold]")
        else:
            # json.dumps shows the default the way the model sees it ("." -> ".")
            default = json.dumps(prop.get("default"), ensure_ascii=False)
            parts.append(f"[dim]{escape(name)}={escape(default)}[/dim]")
    return ", ".join(parts) or "[dim]-[/dim]"


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
def render_error(console: Console, message: str) -> None:
    console.print(f"[bold red]Error:[/bold red] {escape(message)}")


def render_tools_that_ran(console: Console, messages: Sequence[ChatMessage]) -> None:
    """After MaxIterationsExceededError: list the tools the model called."""
    names = [
        call.name
        for message in messages
        if message.tool_calls
        for call in message.tool_calls
    ]
    if names:
        console.print(f"[dim]Tools called before stopping: {', '.join(names)}[/dim]")


def render_unexpected_error(
    console: Console, exc: BaseException, *, debug: bool
) -> None:
    """A bug: short message normally, full traceback with ``--debug``."""
    if debug:
        # show_locals=False: local variables can contain secrets such as the
        # raw API key, and tracebacks get pasted into bug reports.
        console.print(
            Traceback.from_exception(
                type(exc), exc, exc.__traceback__, show_locals=False
            )
        )
    console.print(
        f"[bold red]Unexpected error:[/bold red] "
        f"{escape(type(exc).__name__)}: {escape(str(exc))}"
    )
    if not debug:
        console.print("[dim]Run the command again with --debug for details.[/dim]")


def render_notice(console: Console, message: str) -> None:
    console.print(f"[yellow]{escape(message)}[/yellow]")


# --------------------------------------------------------------------------- #
# Version
# --------------------------------------------------------------------------- #
def render_version(console: Console, rows: Sequence[tuple[str, str]]) -> None:
    # A borderless two-column table lines the values up neatly.
    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    table.add_column(style="bold")
    table.add_column()
    for label, value in rows:
        table.add_row(escape(label), escape(value))
    console.print(table)


# --------------------------------------------------------------------------- #
# Chat session
# --------------------------------------------------------------------------- #
def render_chat_banner(console: Console, *, model: str, base_url: str) -> None:
    console.print(
        Panel(
            f"Model [bold]{escape(model)}[/bold] at {escape(base_url)}\n"
            "Type your question, or [bold]/help[/bold] for commands.",
            title="[bold cyan]Lightspeed chat[/bold cyan]",
            title_align="left",
            border_style="cyan",
        )
    )


def render_chat_help(console: Console) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    table.add_column(style="bold cyan")
    table.add_column()
    for command, description in CHAT_COMMANDS.items():
        table.add_row(command, description)
    console.print(table)


def render_history(console: Console, messages: Sequence[ChatMessage]) -> None:
    questions = [m.content or "" for m in messages if m.role == "user"]
    if not questions:
        console.print("[dim]No questions yet.[/dim]")
        return
    for number, question in enumerate(questions, start=1):
        shown = question if len(question) <= 70 else question[:70] + "..."
        console.print(f"[dim]{number:>3}.[/dim] {escape(shown)}")
