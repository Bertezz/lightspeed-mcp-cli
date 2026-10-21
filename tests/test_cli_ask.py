"""Tests for the ``lightspeed ask`` command.

Learning notes
--------------
The command creates a real ``AssistantClient``, which would talk to a real
server. Here we swap the *class* for a small fake with ``mocker.patch``: the
command then builds our fake instead. The fake behaves like the real thing
from the command's point of view (``async with``, ``chat()``, events).
"""

from types import TracebackType
from typing import Any, Self

import pytest
from pytest_mock import MockerFixture
from typer.testing import CliRunner

from lightspeed_cli.client import ToolEvent, ToolFinished, ToolStarted
from lightspeed_cli.exceptions import LLMConnectionError
from lightspeed_cli.main import app
from lightspeed_cli.schemas import (
    ChatMessage,
    ChatResult,
    TokenUsage,
    ToolExecutionResult,
)

runner = CliRunner()


class FakeAssistant:
    """Stands in for AssistantClient inside the ``ask`` command."""

    result = ChatResult(
        reply="Your CPU is at **12%**.",
        messages=(ChatMessage.user("q"), ChatMessage.assistant("a")),
        usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
        iterations=2,
    )
    error: Exception | None = None

    def __init__(self, *, on_event: Any) -> None:
        self.on_event = on_event

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        pass

    async def chat(self, prompt: str) -> ChatResult:
        if self.error is not None:
            raise self.error
        events: list[ToolEvent] = [
            ToolStarted("c1", "get_system_metrics", {}),
            ToolFinished("c1", "get_system_metrics", ToolExecutionResult.ok(1), 0.4),
            ToolStarted("c2", "inspect_git_status", {}),
            ToolFinished(
                "c2", "inspect_git_status", ToolExecutionResult.fail("no git"), 0.1
            ),
        ]
        for event in events:
            self.on_event(event)
        return self.result


@pytest.fixture
def fake_assistant(mocker: MockerFixture) -> type[FakeAssistant]:
    mocker.patch("lightspeed_cli.main.AssistantClient", FakeAssistant)
    FakeAssistant.error = None
    return FakeAssistant


def test_ask_prints_answer_and_tool_progress(
    fake_assistant: type[FakeAssistant],
) -> None:
    result = runner.invoke(app, ["ask", "How busy is my CPU?"])

    assert result.exit_code == 0
    assert "Your CPU is at 12%." in result.output  # Markdown ** rendered away
    assert "> running get_system_metrics ..." in result.output
    assert "> get_system_metrics ok (0.4s)" in result.output
    assert "> inspect_git_status failed (0.1s)" in result.output
    assert "2 request(s), 0 tool call(s), 30 tokens" in result.output


def test_ask_warns_about_cut_off_answers(
    fake_assistant: type[FakeAssistant], mocker: MockerFixture
) -> None:
    cut_off = fake_assistant.result.model_copy(update={"finish_reason": "length"})
    mocker.patch.object(fake_assistant, "result", cut_off)

    result = runner.invoke(app, ["ask", "Long story please"])

    assert "the answer was cut off" in result.output


def test_ask_shows_friendly_errors(fake_assistant: type[FakeAssistant]) -> None:
    fake_assistant.error = LLMConnectionError("Could not reach the model server.")

    result = runner.invoke(app, ["ask", "Hi"])

    assert result.exit_code == 1
    assert "Error: Could not reach the model server." in result.output
    assert "Traceback" not in result.output


def test_ask_without_api_key_explains_the_fix() -> None:
    # No fake here: the real client refuses to start without a key for OpenAI.
    result = runner.invoke(app, ["ask", "Hi"])

    assert result.exit_code == 1
    assert "OPENAI_API_KEY is not set" in result.output
