"""CLI entry point for ``lightspeed``.

Learning notes
--------------
``pyproject.toml`` maps the terminal command ``lightspeed`` to the ``app`` object
in this file (``[project.scripts]``). Typer turns ordinary functions into
commands: the function name becomes the command name, its parameters become
arguments and options, and its docstring becomes the ``--help`` text.

Command tree::

    lightspeed version
    lightspeed ask "QUESTION"
    lightspeed tools list [--json]
    lightspeed tools run NAME [ARGUMENTS_JSON]
"""

import asyncio
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.table import Table

from lightspeed_cli import __version__
from lightspeed_cli.client import AssistantClient, ToolEvent, ToolStarted
from lightspeed_cli.exceptions import LightspeedError
from lightspeed_cli.schemas import ChatResult, ToolCall
from lightspeed_cli.tools import registry

app = typer.Typer(
    name="lightspeed",
    help="Lightspeed MCP command-line interface.",
    no_args_is_help=True,  # plain `lightspeed` shows help instead of nothing
)
# A sub-application: its commands live under `lightspeed tools ...`.
tools_app = typer.Typer(
    help="Inspect and run the assistant's tools.", no_args_is_help=True
)
app.add_typer(tools_app, name="tools")

# Rich consoles print with colours and formatting. Normal output goes to stdout;
# errors go to stderr, so `lightspeed tools run ... > out.json` captures only
# the result and errors still appear on screen.
console = Console()
err_console = Console(stderr=True)


# @app.callback() marks this function as the *group* behind all subcommands.
# It does nothing itself; its docstring is the top-level help text. Without it,
# Typer would treat a single-command app as that one command.
@app.callback()
def main() -> None:
    """Lightspeed MCP command-line interface."""


@app.command()
def version() -> None:
    """Show the installed version."""
    console.print(f"lightspeed-mcp-cli {__version__}")


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="What you want to know.")],
) -> None:
    """Ask the assistant a one-off question; it may use tools to answer."""
    try:
        # Typer commands are sync functions; asyncio.run() starts an event loop,
        # runs the async code to completion, and closes the loop again.
        result = asyncio.run(_ask(question))
    except LightspeedError as exc:
        # Our own exceptions carry messages written for users: show just that,
        # not a traceback.
        err_console.print(f"[bold red]Error:[/bold red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc

    # The answer goes to stdout; tool progress and notes went to stderr. So
    # `lightspeed ask "..." > answer.md` saves only the answer.
    console.print(Markdown(result.reply))
    if result.finish_reason == "length":
        err_console.print("[yellow]Note: the answer was cut off (length limit).[/]")
    err_console.print(
        f"[dim]{result.iterations} request(s), {len(result.tool_runs)} tool "
        f"call(s), {result.usage.total_tokens} tokens[/dim]"
    )


async def _ask(question: str) -> ChatResult:
    # ``async with`` closes the client's network connections when done.
    async with AssistantClient(on_event=_show_tool_event) as client:
        return await client.chat(question)


def _show_tool_event(event: ToolEvent) -> None:
    """Print one line per tool start/finish, so a slow tool is visible."""
    if isinstance(event, ToolStarted):
        err_console.print(f"[dim]> running {escape(event.name)} ...[/dim]")
        return
    outcome = "ok" if event.result.status == "success" else "[red]failed[/red]"
    err_console.print(
        f"[dim]> {escape(event.name)} {outcome} ({event.duration_seconds:.1f}s)[/dim]"
    )


# ``Annotated[type, typer.Option(...)]`` is how modern Typer attaches CLI
# details (flag name, help text) to a parameter without changing its type.
@tools_app.command("list")
def list_tools(
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Print the schemas exactly as sent to the AI."),
    ] = False,
) -> None:
    """List the tools the assistant can use."""
    if as_json:
        console.print_json(data=registry.get_openai_tools())
        return

    table = Table(title="Available tools")
    table.add_column("Name", style="bold cyan", no_wrap=True)
    table.add_column("Description")
    table.add_column("Parameters")
    for definition in registry.definitions():
        parameters = ", ".join(definition.parameters.properties) or "-"
        # Rich treats [square brackets] as style markup. escape() makes text
        # we did not write ourselves print literally.
        table.add_row(definition.name, escape(definition.description), parameters)
    console.print(table)


@tools_app.command("run")
def run_tool(
    name: Annotated[str, typer.Argument(help="Tool name, e.g. get_system_metrics.")],
    arguments: Annotated[
        str,
        typer.Argument(
            help="Arguments as a JSON object, e.g. '{\"cpu_interval\": 1}'."
        ),
    ] = "{}",
) -> None:
    """Run one tool by hand, the same way the assistant would."""
    # Reuse ToolCall: it already knows how to parse a JSON *string* of
    # arguments and to check the tool name. This is exactly the path the AI's
    # requests will take later.
    try:
        call = ToolCall.model_validate(
            {"id": "cli", "name": name, "arguments": arguments}
        )
    except ValidationError as exc:
        problems = "; ".join(error["msg"] for error in exc.errors())
        # BadParameter makes Typer print a usage error and exit with code 2.
        raise typer.BadParameter(problems) from exc

    # Typer commands are normal (sync) functions, so use the sync wrapper.
    result = registry.execute_sync(call.name, call.arguments)
    if result.status == "error":
        err_console.print(f"[bold red]Error:[/bold red] {escape(result.error or '')}")
        # A non-zero exit code tells scripts and shells that the command failed.
        raise typer.Exit(code=1)
    console.print_json(data=result.output)
