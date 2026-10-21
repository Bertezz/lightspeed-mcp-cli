"""Tests for the rendering helpers in ``lightspeed_cli.ui``.

Learning notes
--------------
Every ui function draws on a ``Console`` passed in as an argument. Here we pass
consoles that write into a ``StringIO`` (an in-memory text file), so a test can
read back exactly what would have appeared on screen.

``force_terminal=True`` makes Rich believe it is talking to a real terminal
(to test the panel); ``color_system=None`` leaves out colour codes so the text
is easy to compare.
"""

from io import StringIO

from rich.console import Console

from lightspeed_cli import ui
from lightspeed_cli.client import ToolFinished, ToolStarted
from lightspeed_cli.schemas import (
    ChatMessage,
    ToolDefinition,
    ToolExecutionResult,
    ToolParameterSchema,
)
from tests.conftest import make_result


def make_console(*, terminal: bool) -> tuple[Console, StringIO]:
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=terminal, color_system=None, width=80)
    return console, buffer


def test_answer_in_a_terminal_is_a_panel_with_rendered_markdown() -> None:
    console, buffer = make_console(terminal=True)

    ui.render_answer(console, make_result("q", reply="CPU is **busy**"), model="m1")

    output = buffer.getvalue()
    assert "Lightspeed" in output
    assert "m1" in output
    assert "CPU is busy" in output  # Markdown rendered: no ** left
    assert "╭" in output  # the panel's rounded corner


def test_answer_when_piped_is_raw_markdown() -> None:
    console, buffer = make_console(terminal=False)

    ui.render_answer(console, make_result("q", reply="CPU is **busy**"), model="m1")

    assert buffer.getvalue() == "CPU is **busy**\n"


def test_tool_lines_escape_markup() -> None:
    # "[bold]" in tool output must be shown literally, not used as a style.
    console, buffer = make_console(terminal=False)
    failed = ToolExecutionResult.fail("index [bold]0[/bold] out of range")

    ui.render_tool_started(console, ToolStarted("c1", "t", {"q": "[red]x"}))
    ui.render_tool_finished(console, ToolFinished("c1", "t", failed, 1.25), debug=True)

    output = buffer.getvalue()
    assert '> t called with {"q": "[red]x"}' in output
    assert "> t failed (1.2s)" in output
    assert "index [bold]0[/bold] out of range" in output


def test_debug_output_is_shortened() -> None:
    console, buffer = make_console(terminal=False)
    big = ToolExecutionResult.ok("y" * 1000)

    ui.render_tool_finished(console, ToolFinished("c1", "t", big, 0.1), debug=True)

    output = buffer.getvalue()
    assert "y" * ui.MAX_DEBUG_OUTPUT not in output.replace("\n", "")
    assert "..." in output


def test_tools_table_marks_required_parameters() -> None:
    console, buffer = make_console(terminal=True)
    definition = ToolDefinition(
        name="search",
        description="Find [things].",
        parameters=ToolParameterSchema(
            properties={
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 5},
            },
            required=("query",),
        ),
    )
    no_params = ToolDefinition(name="ping", description="Ping.")

    ui.render_tools_table(console, [definition, no_params])

    output = buffer.getvalue()
    assert "Find [things]." in output
    assert "query, limit=5" in output
    assert "Bold parameters are required" in output


def test_history_when_empty() -> None:
    console, buffer = make_console(terminal=False)

    ui.render_history(console, ())

    assert "No questions yet." in buffer.getvalue()


def test_tools_that_ran_prints_nothing_without_tool_calls() -> None:
    console, buffer = make_console(terminal=False)

    ui.render_tools_that_ran(console, [ChatMessage.user("hi")])

    assert buffer.getvalue() == ""
