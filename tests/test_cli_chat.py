"""Tests for the interactive ``lightspeed chat`` command.

Learning notes
--------------
``runner.invoke(app, ["chat"], input="hello\\nexit\\n")`` feeds text to the
command's standard input, as if typed: each ``\\n`` is pressing Enter. When the
input runs out, reading another line raises EOFError, exactly like pressing
Ctrl+D in a real terminal.
"""

import pytest
from typer.testing import CliRunner

from lightspeed_cli.exceptions import LLMConnectionError
from lightspeed_cli.main import app
from lightspeed_cli.schemas import ChatMessage
from tests.conftest import FakeAssistantControl, make_result

runner = CliRunner()


def chat(text: str, *args: str) -> tuple[int, str]:
    result = runner.invoke(app, ["chat", *args], input=text)
    return result.exit_code, result.output


@pytest.mark.parametrize("word", ["exit", "quit", "/exit", "EXIT"])
def test_exit_words_end_the_session(
    fake_assistant: FakeAssistantControl, word: str
) -> None:
    code, output = chat(f"{word}\n")

    assert code == 0
    assert "Lightspeed chat" in output  # the banner
    assert "Bye! 0 tokens used this session." in output
    assert fake_assistant.prompts == []
    assert fake_assistant.closed == 1  # the client was closed on the way out


def test_end_of_input_ends_the_session(fake_assistant: FakeAssistantControl) -> None:
    code, output = chat("")  # no input at all: like pressing Ctrl+D at once

    assert code == 0
    assert "Bye!" in output


def test_history_carries_over_between_turns(
    fake_assistant: FakeAssistantControl,
) -> None:
    code, output = chat("I am Ada\nWho am I?\nexit\n")

    assert code == 0
    assert fake_assistant.prompts == ["I am Ada", "Who am I?"]
    assert fake_assistant.histories[0] == ()
    assert fake_assistant.histories[1] == (
        ChatMessage.user("I am Ada"),
        ChatMessage.assistant("Echo: I am Ada"),
    )
    assert "Echo: Who am I?" in output
    assert "Bye! 24 tokens used this session." in output  # 2 turns x 12


def test_blank_lines_are_ignored(fake_assistant: FakeAssistantControl) -> None:
    chat("\n   \nhello\nexit\n")

    assert fake_assistant.prompts == ["hello"]


def test_clear_starts_a_new_conversation(
    fake_assistant: FakeAssistantControl,
) -> None:
    _, output = chat("first\n/clear\nsecond\nexit\n")

    assert "Started a new conversation." in output
    assert fake_assistant.histories[1] == ()  # nothing carried over


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("/help", "Forget the conversation"),
        ("/tools", "get_system_metrics"),
        ("/history", "No questions yet."),
        ("/nonsense", "Unknown command /nonsense; type /help."),
    ],
)
def test_slash_commands(
    fake_assistant: FakeAssistantControl, command: str, expected: str
) -> None:
    code, output = chat(f"{command}\nexit\n")

    assert code == 0
    assert expected in output
    assert fake_assistant.prompts == []  # slash commands never reach the model


def test_history_lists_questions(fake_assistant: FakeAssistantControl) -> None:
    long_question = "x" * 100
    _, output = chat(f"first question\n{long_question}\n/history\nexit\n")

    assert "1. first question" in output
    assert "2. " + "x" * 70 + "..." in output  # long questions are shortened


def test_failed_turn_keeps_history_and_session(
    fake_assistant: FakeAssistantControl,
) -> None:
    fake_assistant.outcomes.extend(
        [
            make_result("one"),
            LLMConnectionError("Could not reach the model server"),
        ]
    )

    code, output = chat("one\ntwo\nthree\nexit\n")

    assert code == 0  # the session survived the error
    assert "Error: Could not reach the model server" in output
    # "two" failed, so "three" is sent with the history from "one" only.
    assert fake_assistant.histories[2] == (
        ChatMessage.user("one"),
        ChatMessage.assistant("Echo: one"),
    )


def test_ctrl_c_cancels_only_the_current_answer(
    fake_assistant: FakeAssistantControl,
) -> None:
    fake_assistant.outcomes.extend([make_result("one"), KeyboardInterrupt()])

    code, output = chat("one\ntwo\nthree\nexit\n")

    assert code == 0
    assert "Cancelled; the conversation is unchanged." in output
    assert fake_assistant.prompts == ["one", "two", "three"]
    assert len(fake_assistant.histories[2]) == 2  # only turn "one"


def test_bug_in_a_turn_does_not_end_the_session(
    fake_assistant: FakeAssistantControl,
) -> None:
    fake_assistant.outcomes.append(RuntimeError("kaboom"))

    code, output = chat("one\ntwo\nexit\n")

    assert code == 0
    assert "Unexpected error: RuntimeError: kaboom" in output
    assert "Echo: two" in output


def test_options_are_passed_to_the_client(
    fake_assistant: FakeAssistantControl,
) -> None:
    _, output = chat("exit\n", "--model", "qwen2.5", "-t", "0")

    assert fake_assistant.created == [{"model": "qwen2.5", "temperature": 0.0}]
    assert "Model qwen2.5" in output


def test_start_up_error_exits_with_1() -> None:
    # No fake: without an API key the real client cannot be created.
    code, output = chat("hello\n")

    assert code == 1
    assert "OPENAI_API_KEY is not set" in output
    assert "Lightspeed chat" not in output  # never got as far as the banner
