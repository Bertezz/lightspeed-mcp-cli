"""CLI entry point for ``lightspeed``.

Learning notes
--------------
``pyproject.toml`` maps the terminal command ``lightspeed`` to the ``app`` object
in this file (``[project.scripts]``). Typer turns ordinary functions into
commands: the function name becomes the command name, its parameters become
arguments and options, and its docstring becomes the ``--help`` text.

This file decides *what happens*; ``ui.py`` decides *how it looks*.

Command tree::

    lightspeed [--version]
    lightspeed version
    lightspeed ask "QUESTION" [--model M] [--temperature T] [--json] [--debug]
    lightspeed chat           [--model M] [--temperature T] [--debug]
    lightspeed tools list [--json]
    lightspeed tools run NAME [ARGUMENTS_JSON]

Output streams: answers go to *stdout*; progress, notes and errors go to
*stderr*. So ``lightspeed ask "..." > answer.md`` saves only the answer.

Exit codes: 0 success, 1 our errors, 2 wrong usage (Typer), 130 Ctrl+C.
"""

import asyncio
import platform
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from typing import Annotated

import openai
import typer
from pydantic import ValidationError
from rich.console import Console
from rich.status import Status

from lightspeed_cli import __version__, ui
from lightspeed_cli.client import (
    AssistantClient,
    ToolEvent,
    ToolStarted,
    trim_history,
)
from lightspeed_cli.config import Settings, load_settings
from lightspeed_cli.exceptions import LightspeedError, MaxIterationsExceededError
from lightspeed_cli.logging_setup import configure_logging
from lightspeed_cli.schemas import ChatMessage, ChatResult, ToolCall
from lightspeed_cli.tools import registry

app = typer.Typer(
    name="lightspeed",
    help="Lightspeed: a command-line assistant that can inspect your system.",
    no_args_is_help=True,  # plain `lightspeed` shows help instead of nothing
    # Never print local variables in crash reports: they can hold secrets such
    # as the raw API key. (Newer Typer defaults to False; older versions, which
    # our dependency range still allows, default to True.)
    pretty_exceptions_show_locals=False,
)
# A sub-application: its commands live under `lightspeed tools ...`.
tools_app = typer.Typer(
    help="Inspect and run the assistant's tools.", no_args_is_help=True
)
app.add_typer(tools_app, name="tools")

console = Console()  # stdout: answers and requested output
err_console = Console(stderr=True)  # stderr: progress, notes, errors

EXIT_ERROR = 1
EXIT_INTERRUPTED = 130  # shell convention: 128 + signal number 2 (SIGINT)
EXIT_WORDS = {"exit", "quit", "/exit", "/quit"}

# --------------------------------------------------------------------------- #
# Options shared by `ask` and `chat`. ``Annotated[type, typer.Option(...)]``
# attaches the CLI details to a type; defining them once avoids repeating them.
# --------------------------------------------------------------------------- #
ModelOption = Annotated[
    str | None,
    typer.Option("--model", "-m", help="Model to use instead of DEFAULT_MODEL."),
]
TemperatureOption = Annotated[
    float | None,
    typer.Option(
        "--temperature",
        "-t",
        min=0.0,
        max=2.0,
        help="Randomness, 0-2 (lower = more predictable). Default: the server's.",
    ),
]
DebugOption = Annotated[
    bool,
    typer.Option("--debug", help="Show tool schemas, arguments, results and logs."),
]


# --------------------------------------------------------------------------- #
# Root callback and version
# --------------------------------------------------------------------------- #
def _version_callback(value: bool) -> None:
    # Called by Typer as soon as --version is seen (is_eager=True means "before
    # anything else"), then we exit, so no command is required.
    if value:
        ui.render_version(console, _version_rows())
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show version information and exit.",
        ),
    ] = False,
) -> None:
    """Lightspeed: a command-line assistant that can inspect your system."""


@app.command()
def version() -> None:
    """Show version and environment information."""
    ui.render_version(console, _version_rows())


def _version_rows() -> list[tuple[str, str]]:
    rows = [
        ("lightspeed-mcp-cli", __version__),
        ("Python", platform.python_version()),
        (
            "Platform",
            f"{platform.system()} {platform.release()} ({platform.machine()})",
        ),
        ("openai SDK", openai.__version__),
    ]
    # Settings are shown for convenience; broken settings must not stop the
    # version from being printed (you may need it to report the problem).
    # Note: the API key is deliberately never shown.
    try:
        settings = Settings()
    except ValidationError as exc:
        rows.append(("Settings", f"invalid ({exc.error_count()} problem(s))"))
    else:
        rows.append(("Model", settings.default_model))
        rows.append(("Server", settings.base_url_str() or "OpenAI (default)"))
    return rows


# --------------------------------------------------------------------------- #
# Error handling shared by all commands
# --------------------------------------------------------------------------- #
def _report_error(exc: BaseException, *, debug: bool) -> int:
    """Show ``exc`` the friendly way and return the exit code to use."""
    if isinstance(exc, KeyboardInterrupt):
        ui.render_notice(err_console, "Interrupted.")
        return EXIT_INTERRUPTED
    if isinstance(exc, LightspeedError):
        # Our own errors carry messages written for users: no traceback.
        ui.render_error(err_console, str(exc))
        if isinstance(exc, MaxIterationsExceededError):
            ui.render_tools_that_ran(err_console, exc.messages)
        return EXIT_ERROR
    # Anything else is a bug in our code (or a library's).
    ui.render_unexpected_error(err_console, exc, debug=debug)
    return EXIT_ERROR


@contextmanager
def _friendly_errors(*, debug: bool) -> Iterator[None]:
    """Turn exceptions raised inside the ``with`` block into clean exits.

    ``@contextmanager`` turns a generator into something usable with ``with``:
    the code before ``yield`` runs on entry, the block runs at the ``yield``,
    and exceptions from the block are re-raised at the ``yield``, where our
    ``except`` clauses can catch them.
    """
    try:
        yield
    except typer.Exit:
        # typer.Exit is itself an Exception subclass: let it through untouched,
        # or the ``except Exception`` below would report it as a bug.
        raise
    except (Exception, KeyboardInterrupt) as exc:
        raise typer.Exit(code=_report_error(exc, debug=debug)) from exc


def _start(debug: bool) -> Settings:
    """Common start-up for commands that talk to the model."""
    settings = load_settings()  # invalid settings -> ConfigurationError
    configure_logging(settings.log_level, debug=debug, console=err_console)
    if debug:
        ui.render_debug_tools(err_console, registry.get_openai_tools())
    return settings


class _ToolProgress:
    """Receives tool events from the client and shows them while a spinner runs.

    A class with ``__call__`` can be used like a function, so an instance can be
    passed as the client's ``on_event`` callback while still remembering state
    (here: the current spinner, which changes every turn in a chat).
    """

    def __init__(self, *, debug: bool) -> None:
        self.debug = debug
        self.status: Status | None = None

    def __call__(self, event: ToolEvent) -> None:
        if isinstance(event, ToolStarted):
            if self.status is not None:
                self.status.update(f"Running {event.name}...")
            if self.debug:
                ui.render_tool_started(err_console, event)
            return
        # Printing while a spinner is active is fine: Rich prints the line
        # above the spinner and keeps the spinner at the bottom.
        ui.render_tool_finished(err_console, event, debug=self.debug)
        if self.status is not None:
            self.status.update("Thinking...")

    @contextmanager
    def spinner(self) -> Iterator[None]:
        # When stderr is not a terminal (pipes, tests), Rich shows nothing.
        with err_console.status("Thinking...") as status:
            self.status = status
            try:
                yield
            finally:
                self.status = None


# --------------------------------------------------------------------------- #
# ask: one question, one answer
# --------------------------------------------------------------------------- #
@app.command()
def ask(
    prompt: Annotated[str, typer.Argument(help="What you want to know.")],
    model: ModelOption = None,
    temperature: TemperatureOption = None,
    debug: DebugOption = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the full result as JSON.")
    ] = False,
) -> None:
    """Ask the assistant a one-off question; it may use tools to answer."""
    with _friendly_errors(debug=debug):
        settings = _start(debug)
        progress = _ToolProgress(debug=debug)
        client = AssistantClient(
            settings, model=model, temperature=temperature, on_event=progress
        )
        with progress.spinner():
            # Typer commands are sync functions; asyncio.run() starts an event
            # loop, runs the coroutine to completion, and closes the loop.
            result = asyncio.run(_chat_once(client, prompt))

    if as_json:
        console.print_json(result.model_dump_json())
    else:
        ui.render_answer(console, result, model=client.model)
    ui.render_stats(err_console, result)


async def _chat_once(client: AssistantClient, prompt: str) -> ChatResult:
    # ``async with`` closes the client's network connections when done.
    async with client:
        return await client.chat(prompt)


# --------------------------------------------------------------------------- #
# chat: an interactive session (a REPL: Read, Evaluate, Print, Loop)
# --------------------------------------------------------------------------- #
@app.command()
def chat(
    model: ModelOption = None,
    temperature: TemperatureOption = None,
    debug: DebugOption = False,
) -> None:
    """Start an interactive conversation (type /help inside for commands)."""
    _enable_line_editing()
    progress = _ToolProgress(debug=debug)
    with _friendly_errors(debug=debug):  # start-up errors end the command
        settings = _start(debug)
        client = AssistantClient(
            settings, model=model, temperature=temperature, on_event=progress
        )

    ui.render_chat_banner(console, model=client.model, base_url=client.base_url)
    # asyncio.Runner keeps ONE event loop for the whole session. The client's
    # network connections belong to the loop that created them, so creating a
    # new loop per question (asyncio.run each time) would break them.
    with asyncio.Runner() as runner:
        try:
            total_tokens = _chat_loop(runner, client, progress, debug=debug)
        finally:
            runner.run(client.aclose())
    err_console.print(f"[dim]Bye! {total_tokens} tokens used this session.[/dim]")


def _chat_loop(
    runner: asyncio.Runner,
    client: AssistantClient,
    progress: _ToolProgress,
    *,
    debug: bool,
) -> int:
    """Read questions until the user leaves; return the tokens used."""
    history: tuple[ChatMessage, ...] = ()
    total_tokens = 0
    while True:
        try:
            line = console.input("[bold green]you>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            # EOFError: Ctrl+D (or the end of piped input).
            # KeyboardInterrupt: Ctrl+C while nobody is waiting for an answer.
            console.print()
            return total_tokens

        if not line:
            continue
        if line.lower() in EXIT_WORDS:
            return total_tokens
        if line.startswith("/"):
            history = _run_slash_command(line, history)
            continue

        try:
            with progress.spinner():
                result = runner.run(client.chat(line, history))
        except KeyboardInterrupt:
            # Ctrl+C while waiting: cancel this answer only, keep the session.
            ui.render_notice(err_console, "Cancelled; the conversation is unchanged.")
            continue
        except Exception as exc:
            # The failed turn is simply not added to the history, so the
            # conversation stays valid and the user can just try again.
            _report_error(exc, debug=debug)
            continue

        history = trim_history(result.messages)
        total_tokens += result.usage.total_tokens
        ui.render_answer(console, result, model=client.model)
        ui.render_stats(err_console, result)


def _run_slash_command(
    line: str, history: tuple[ChatMessage, ...]
) -> tuple[ChatMessage, ...]:
    """Handle a /command and return the (possibly changed) history."""
    command = line.split()[0].lower()
    handlers: dict[str, Callable[[], None]] = {
        "/help": lambda: ui.render_chat_help(console),
        "/tools": lambda: ui.render_tools_table(console, registry.definitions()),
        "/history": lambda: ui.render_history(console, history),
    }
    if command == "/clear":
        ui.render_notice(console, "Started a new conversation.")
        return ()
    handler = handlers.get(command)
    if handler is None:
        ui.render_notice(console, f"Unknown command {command}; type /help.")
    else:
        handler()
    return history


def _enable_line_editing() -> None:
    # Importing the standard ``readline`` module upgrades input(): arrow keys
    # move the cursor and Up/Down recall earlier lines. It does not exist on
    # Windows, where we simply do without (suppress ignores the ImportError).
    with suppress(ImportError):
        import readline  # noqa: F401


# --------------------------------------------------------------------------- #
# tools: inspect and run tools by hand
# --------------------------------------------------------------------------- #
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
    ui.render_tools_table(console, registry.definitions())


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
    # arguments and to check the tool name, exactly as for the AI's requests.
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
        ui.render_error(err_console, result.error or "")
        raise typer.Exit(code=EXIT_ERROR)
    console.print_json(data=result.output)
