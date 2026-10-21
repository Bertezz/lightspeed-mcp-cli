"""Tests for the ``lightspeed ask`` command.

Learning notes
--------------
The ``fake_assistant`` fixture (see conftest.py) replaces the real client, so
these tests check only what the *command* does: options, output, exit codes.
The client itself is tested in test_client.py.

``CliRunner`` output is not a terminal, so ``ask`` prints the answer as plain
text (no panel) and Rich shows no spinner; ui.py's panel mode is tested in
test_ui.py instead.
"""

import json

import pytest
import typer
from typer.testing import CliRunner

from lightspeed_cli.client import ToolEvent, ToolFinished, ToolStarted
from lightspeed_cli.exceptions import (
    ConfigurationError,
    LightspeedError,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    MaxIterationsExceededError,
)
from lightspeed_cli.main import _friendly_errors, _ToolProgress, app
from lightspeed_cli.schemas import ChatMessage, ToolCall, ToolExecutionResult
from tests.conftest import FakeAssistantControl, make_result

runner = CliRunner()

CPU_EVENTS: list[ToolEvent] = [
    ToolStarted("c1", "get_system_metrics", {"cpu_interval": 0.1}),
    ToolFinished("c1", "get_system_metrics", ToolExecutionResult.ok({"cpu": 12}), 0.4),
    ToolStarted("c2", "inspect_git_status", {}),
    ToolFinished("c2", "inspect_git_status", ToolExecutionResult.fail("no git"), 0.1),
]


def test_ask_prints_answer_and_stats(fake_assistant: FakeAssistantControl) -> None:
    fake_assistant.outcomes.append(make_result("q", reply="Your CPU is at **12%**."))

    result = runner.invoke(app, ["ask", "How busy is my CPU?"])

    assert result.exit_code == 0
    # Not a terminal: the raw Markdown is printed, ready to save to a file.
    assert "Your CPU is at **12%**." in result.output
    assert "1 request(s), 0 tool call(s), 12 tokens" in result.output
    assert fake_assistant.prompts == ["How busy is my CPU?"]
    assert fake_assistant.closed == 1  # `async with` closed the client


def test_ask_shows_tool_progress(fake_assistant: FakeAssistantControl) -> None:
    fake_assistant.events.extend(CPU_EVENTS)

    result = runner.invoke(app, ["ask", "status?"])

    assert "> get_system_metrics ok (0.4s)" in result.output
    assert "> inspect_git_status failed (0.1s)" in result.output
    assert "called with" not in result.output  # arguments only with --debug


def test_ask_debug_shows_schemas_arguments_and_results(
    fake_assistant: FakeAssistantControl,
) -> None:
    fake_assistant.events.extend(CPU_EVENTS)

    result = runner.invoke(app, ["ask", "status?", "--debug"])

    assert "tool schema(s) to the model" in result.output
    assert '"name": "get_system_metrics"' in result.output  # the schema JSON
    assert '> get_system_metrics called with {"cpu_interval": 0.1}' in result.output
    assert '{"cpu": 12}' in result.output  # the tool result
    assert "no git" in result.output  # the tool error


def test_ask_passes_model_and_temperature(fake_assistant: FakeAssistantControl) -> None:
    result = runner.invoke(app, ["ask", "hi", "-m", "llama3.1", "--temperature", "0.2"])

    assert result.exit_code == 0
    assert fake_assistant.created == [{"model": "llama3.1", "temperature": 0.2}]


@pytest.mark.parametrize("value", ["-0.1", "2.5", "warm"])
def test_ask_rejects_invalid_temperature(
    fake_assistant: FakeAssistantControl, value: str
) -> None:
    result = runner.invoke(app, ["ask", "hi", "--temperature", value])

    assert result.exit_code == 2  # a usage error, reported by Typer
    assert fake_assistant.created == []  # nothing was sent


def test_ask_json_output(fake_assistant: FakeAssistantControl) -> None:
    result = runner.invoke(app, ["ask", "hi", "--json"])

    # The JSON is on stdout; the stats line went to stderr.
    data = json.loads(result.stdout)
    assert data["reply"] == "Echo: hi"
    assert data["usage"]["total_tokens"] == 12


@pytest.mark.parametrize(
    "error",
    [
        ConfigurationError("OPENAI_API_KEY is not set"),
        LLMConnectionError("Could not reach the model server"),
        LLMTimeoutError("did not answer in time"),
        LLMRateLimitError("rate-limiting requests"),
        LLMAuthenticationError("rejected the API key"),
        LLMResponseError("model 'x' not found"),
    ],
    ids=lambda error: type(error).__name__,
)
def test_ask_reports_our_errors_without_traceback(
    fake_assistant: FakeAssistantControl, error: LightspeedError
) -> None:
    fake_assistant.outcomes.append(error)

    result = runner.invoke(app, ["ask", "hi"])

    assert result.exit_code == 1
    assert f"Error: {error}" in result.output
    assert "Traceback" not in result.output


def test_ask_max_iterations_lists_tools_that_ran(
    fake_assistant: FakeAssistantControl,
) -> None:
    calls = [ToolCall(id="c1", name="get_system_metrics")]
    messages = (
        ChatMessage.user("loop"),
        ChatMessage.assistant(tool_calls=calls),
        ChatMessage.tool("c1", "{}"),
    )
    fake_assistant.outcomes.append(
        MaxIterationsExceededError(iterations=5, messages=messages)
    )

    result = runner.invoke(app, ["ask", "loop"])

    assert result.exit_code == 1
    assert "still calling tools after 5 rounds" in result.output
    assert "Tools called before stopping: get_system_metrics" in result.output


def test_ask_reports_bugs_briefly(fake_assistant: FakeAssistantControl) -> None:
    fake_assistant.outcomes.append(RuntimeError("kaboom"))

    result = runner.invoke(app, ["ask", "hi"])

    assert result.exit_code == 1
    assert "Unexpected error: RuntimeError: kaboom" in result.output
    assert "--debug" in result.output
    assert "Traceback" not in result.output


def test_ask_debug_shows_traceback_for_bugs(
    fake_assistant: FakeAssistantControl,
) -> None:
    fake_assistant.outcomes.append(RuntimeError("kaboom"))

    result = runner.invoke(app, ["ask", "hi", "--debug"])

    assert result.exit_code == 1
    assert "Traceback" in result.output
    assert "Unexpected error: RuntimeError: kaboom" in result.output


def test_ask_ctrl_c_exits_with_130(fake_assistant: FakeAssistantControl) -> None:
    fake_assistant.outcomes.append(KeyboardInterrupt())

    result = runner.invoke(app, ["ask", "hi"])

    assert result.exit_code == 130
    assert "Interrupted." in result.output


def test_ask_warns_about_cut_off_answers(fake_assistant: FakeAssistantControl) -> None:
    cut_off = make_result("q").model_copy(update={"finish_reason": "length"})
    fake_assistant.outcomes.append(cut_off)

    result = runner.invoke(app, ["ask", "Long story please"])

    assert "the answer was cut off" in result.output


def test_ask_with_invalid_settings(
    fake_assistant: FakeAssistantControl, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOG_LEVEL", "TRACE")

    result = runner.invoke(app, ["ask", "hi"])

    assert result.exit_code == 1
    # Rich wraps long lines at the terminal width; compare without line breaks.
    output = " ".join(result.output.split())
    assert "Invalid settings" in output
    assert "LIGHTSPEED_LOG_LEVEL / LOG_LEVEL" in output


def test_ask_without_api_key_explains_the_fix() -> None:
    # No fake here: the real client refuses to start without a key for OpenAI.
    result = runner.invoke(app, ["ask", "Hi"])

    assert result.exit_code == 1
    assert "OPENAI_API_KEY is not set" in result.output


# --------------------------------------------------------------------------- #
# Internals of main.py, tested directly
# --------------------------------------------------------------------------- #
def test_friendly_errors_lets_typer_exit_through() -> None:
    # typer.Exit is an Exception subclass. If _friendly_errors treated it as
    # a bug, every deliberate `raise typer.Exit(...)` would print
    # "Unexpected error" and lose its exit code.
    with pytest.raises(typer.Exit) as exc_info, _friendly_errors(debug=False):
        raise typer.Exit(code=7)

    assert exc_info.value.exit_code == 7


def test_tool_progress_without_spinner(capsys: pytest.CaptureFixture[str]) -> None:
    # Events can arrive when no spinner is active; nothing must crash.
    progress = _ToolProgress(debug=False)

    for event in CPU_EVENTS:
        progress(event)

    assert "> get_system_metrics ok (0.4s)" in capsys.readouterr().err
