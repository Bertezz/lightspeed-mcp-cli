"""Tests for the assistant client and its agent loop (``lightspeed_cli.client``).

Learning notes
--------------
These tests never touch the network. Instead of mocking the ``openai`` SDK's
internals, we replace only the very bottom layer: the HTTP transport.
``httpx2.MockTransport`` hands every HTTP request to our ``FakeLLMServer``,
which answers with JSON shaped exactly like the real API.

Everything above that layer is real code: the SDK builds the request, parses
the response into its own types, retries, and turns HTTP 429 into
``RateLimitError``. So the tests prove our client works with the *real* SDK,
and they keep working when SDK internals change.

The fake server also records every request, so tests can check exactly what
was sent to the model (the conversation, the tools, the tool results).
"""

import asyncio
import json
import time
from collections.abc import Callable
from typing import Any

import httpx2
import openai
import pytest
from openai import AsyncOpenAI
from pydantic import HttpUrl, SecretStr

from lightspeed_cli.client import (
    DEFAULT_SYSTEM_PROMPT,
    INVALID_TOOL_NAME,
    PLACEHOLDER_API_KEY,
    AssistantClient,
    ToolEvent,
    ToolFinished,
    ToolStarted,
    _to_openai_messages,
    _translate_openai_error,
    trim_history,
)
from lightspeed_cli.config import Settings
from lightspeed_cli.exceptions import (
    ConfigurationError,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    MaxIterationsExceededError,
)
from lightspeed_cli.schemas import ChatMessage, TokenUsage, ToolCall
from lightspeed_cli.tools import ToolRegistry
from lightspeed_cli.tools import registry as default_registry

# --------------------------------------------------------------------------- #
# The fake model server
# --------------------------------------------------------------------------- #
Reply = httpx2.Response | Exception


class FakeLLMServer:
    """Answers HTTP requests from a queue of prepared replies."""

    def __init__(self) -> None:
        self.replies: list[Reply] = []
        self.requests: list[dict[str, Any]] = []  # parsed JSON bodies

    def queue(self, *replies: Reply) -> None:
        self.replies.extend(replies)

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply  # e.g. a connection error or timeout
        return reply

    def sent_messages(self, index: int = -1) -> list[dict[str, Any]]:
        """The "messages" list of one recorded request (default: the last)."""
        messages: list[dict[str, Any]] = self.requests[index]["messages"]
        return messages


def completion(
    message: dict[str, Any],
    finish_reason: str = "stop",
    usage: tuple[int, int] | None = (10, 5),
) -> httpx2.Response:
    """Build a chat-completion response like the real API returns."""
    body: dict[str, Any] = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": None, **message},
            }
        ],
    }
    if usage is not None:
        prompt, completion_tokens = usage
        body["usage"] = {
            "prompt_tokens": prompt,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt + completion_tokens,
        }
    return httpx2.Response(200, json=body)


def text_reply(text: str, **kwargs: Any) -> httpx2.Response:
    return completion({"content": text}, **kwargs)


def tool_reply(*calls: tuple[str, str, str]) -> httpx2.Response:
    """A reply asking for tools; each call is (id, name, arguments_json)."""
    return completion(
        {
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
                for call_id, name, arguments in calls
            ]
        },
        finish_reason="tool_calls",
    )


def error_reply(status: int, message: str) -> httpx2.Response:
    return httpx2.Response(status, json={"error": {"message": message}})


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def server() -> FakeLLMServer:
    return FakeLLMServer()


@pytest.fixture
def tools() -> ToolRegistry:
    """Small, fast, predictable tools instead of the real system tools."""
    registry = ToolRegistry()

    @registry.register()
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @registry.register()
    def get_weather(city: str) -> str:
        """Return the weather for a city."""
        return f"Sunny in {city}"

    @registry.register()
    async def slow(seconds: float) -> str:
        """Wait, then finish."""
        await asyncio.sleep(seconds)
        return "done"

    return registry


@pytest.fixture
def settings() -> Settings:
    return Settings(
        openai_api_key=SecretStr("test-key"),
        default_model="test-model",
        max_tool_iterations=3,
    )


# A fixture can return a *function*: tests call it to build a client with the
# options they need ("factory fixture").
ClientFactory = Callable[..., AssistantClient]


@pytest.fixture
def make_client(
    server: FakeLLMServer, tools: ToolRegistry, settings: Settings
) -> ClientFactory:
    def factory(**kwargs: Any) -> AssistantClient:
        http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(server.handle))
        fake_openai = AsyncOpenAI(
            api_key="test-key",
            base_url="http://fake.test/v1",
            max_retries=0,  # fail at once instead of retrying (fast tests)
            http_client=http_client,
        )
        kwargs.setdefault("registry", tools)
        return AssistantClient(settings, openai_client=fake_openai, **kwargs)

    return factory


# --------------------------------------------------------------------------- #
# 1. Direct answers
# --------------------------------------------------------------------------- #
async def test_direct_answer(server: FakeLLMServer, make_client: ClientFactory) -> None:
    server.queue(text_reply("Hello! How can I help?"))

    result = await make_client().chat("Hi")

    assert result.reply == "Hello! How can I help?"
    assert result.iterations == 1
    assert result.tool_runs == ()
    assert result.finish_reason == "stop"
    assert result.usage == TokenUsage(
        prompt_tokens=10, completion_tokens=5, total_tokens=15
    )
    assert result.messages == (
        ChatMessage.user("Hi"),
        ChatMessage.assistant("Hello! How can I help?"),
    )


async def test_request_contents(
    server: FakeLLMServer, make_client: ClientFactory
) -> None:
    server.queue(text_reply("ok"))

    result = await make_client(system_prompt="Be brief.").chat("Hi")

    request = server.requests[0]
    assert request["model"] == "test-model"
    assert request["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
    ]
    assert {tool["function"]["name"] for tool in request["tools"]} == {
        "add",
        "get_weather",
        "slow",
    }
    # The system prompt is sent every time but never stored in the history.
    assert all(message.role != "system" for message in result.messages)


async def test_default_system_prompt(
    server: FakeLLMServer, make_client: ClientFactory
) -> None:
    server.queue(text_reply("ok"))

    await make_client().chat("Hi")

    assert server.sent_messages()[0] == {
        "role": "system",
        "content": DEFAULT_SYSTEM_PROMPT,
    }


async def test_empty_registry_sends_no_tools_field(
    server: FakeLLMServer, make_client: ClientFactory
) -> None:
    server.queue(text_reply("ok"))

    # An empty registry must stay empty (not be swapped for the default one).
    await make_client(registry=ToolRegistry()).chat("Hi")

    assert "tools" not in server.requests[0]


async def test_multi_turn_conversation(
    server: FakeLLMServer, make_client: ClientFactory
) -> None:
    server.queue(text_reply("Hi Ada!"), text_reply("You are Ada."))
    client = make_client()

    first = await client.chat("I am Ada.")
    second = await client.chat("Who am I?", first.messages)

    assert server.sent_messages() == [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": "I am Ada."},
        {"role": "assistant", "content": "Hi Ada!"},
        {"role": "user", "content": "Who am I?"},
    ]
    assert len(second.messages) == 4


# --------------------------------------------------------------------------- #
# 2. One tool call
# --------------------------------------------------------------------------- #
async def test_single_tool_call(
    server: FakeLLMServer, make_client: ClientFactory
) -> None:
    server.queue(
        tool_reply(("call_1", "add", '{"a": 2, "b": 3}')),
        text_reply("2 + 3 = 5"),
    )

    result = await make_client().chat("What is 2 + 3?")

    assert result.reply == "2 + 3 = 5"
    assert result.iterations == 2
    assert [(run.name, run.result.output) for run in result.tool_runs] == [("add", 5)]
    assert result.usage.total_tokens == 30  # added up over both requests
    # The second request carries the tool call *and* its result, linked by id.
    assert server.sent_messages()[-2:] == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'},
                }
            ],
        },
        {"role": "tool", "content": "5", "tool_call_id": "call_1"},
    ]


# --------------------------------------------------------------------------- #
# 3. Several tool calls
# --------------------------------------------------------------------------- #
async def test_sequential_tool_calls(
    server: FakeLLMServer, make_client: ClientFactory
) -> None:
    server.queue(
        tool_reply(("call_1", "add", '{"a": 1, "b": 1}')),
        tool_reply(("call_2", "get_weather", '{"city": "Budapest"}')),
        text_reply("1 + 1 = 2, and it is sunny in Budapest."),
    )

    result = await make_client().chat("Two questions")

    assert result.iterations == 3
    assert [run.name for run in result.tool_runs] == ["add", "get_weather"]
    roles = [message.role for message in result.messages]
    assert roles == ["user", "assistant", "tool", "assistant", "tool", "assistant"]


async def test_parallel_tool_calls_run_at_the_same_time(
    server: FakeLLMServer, make_client: ClientFactory
) -> None:
    server.queue(
        tool_reply(
            ("call_a", "slow", '{"seconds": 0.2}'),
            ("call_b", "add", '{"a": 1, "b": 2}'),
            ("call_c", "slow", '{"seconds": 0.2}'),
        ),
        text_reply("All done."),
    )

    started = time.perf_counter()
    result = await make_client().chat("Do three things")
    elapsed = time.perf_counter() - started

    # One after another would take ~0.4s; at the same time, ~0.2s.
    assert elapsed < 0.35
    # Results keep the order of the calls.
    assert [run.call_id for run in result.tool_runs] == ["call_a", "call_b", "call_c"]
    tool_messages = [m for m in result.messages if m.role == "tool"]
    assert [m.tool_call_id for m in tool_messages] == ["call_a", "call_b", "call_c"]


# --------------------------------------------------------------------------- #
# 4. Bad tool calls do not crash the conversation
# --------------------------------------------------------------------------- #
async def test_malformed_arguments_get_a_second_chance(
    server: FakeLLMServer, make_client: ClientFactory
) -> None:
    server.queue(
        tool_reply(("call_1", "add", '{"a": 2, "b":')),  # cut-off JSON
        tool_reply(("call_2", "add", '{"a": 2, "b": 3}')),  # model fixes it
        text_reply("5"),
    )

    result = await make_client().chat("2 + 3?")

    first, second = result.tool_runs
    assert first.result.status == "error"
    assert first.result.error is not None
    assert "not a valid JSON object" in first.result.error
    assert first.arguments == {}  # the broken call is kept, without arguments
    assert second.result.output == 5
    assert result.reply == "5"


@pytest.mark.parametrize(
    ("raw_call", "expected_name", "expected_error"),
    [
        (
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "nope", "arguments": "{}"},
            },
            "nope",
            "Unknown tool 'nope'",
        ),
        (
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "bad name!", "arguments": "{}"},
            },
            INVALID_TOOL_NAME,
            "'bad name!' is not a valid tool name",
        ),
        (
            {"id": "c1", "type": "custom", "custom": {"name": "grep", "input": "x"}},
            "grep",
            "Unsupported tool call type 'custom'",
        ),
    ],
    ids=["unknown-tool", "invalid-name", "custom-type"],
)
async def test_unusable_tool_calls_become_error_results(
    server: FakeLLMServer,
    make_client: ClientFactory,
    raw_call: dict[str, Any],
    expected_name: str,
    expected_error: str,
) -> None:
    server.queue(
        completion({"tool_calls": [raw_call]}, finish_reason="tool_calls"),
        text_reply("Sorry, I could not do that."),
    )

    result = await make_client().chat("Do something")

    (run,) = result.tool_runs  # "unpack exactly one": fails if there are more
    assert run.name == expected_name
    assert run.result.error is not None
    assert expected_error in run.result.error
    assert result.reply == "Sorry, I could not do that."


async def test_missing_tool_call_id_is_generated(
    server: FakeLLMServer, make_client: ClientFactory
) -> None:
    server.queue(tool_reply(("", "add", '{"a": 1, "b": 1}')), text_reply("2"))

    result = await make_client().chat("1 + 1?")

    assert result.tool_runs[0].call_id == "call_1_0"


# --------------------------------------------------------------------------- #
# 5. Endless tool loops are stopped
# --------------------------------------------------------------------------- #
async def test_max_iterations(
    server: FakeLLMServer, make_client: ClientFactory
) -> None:
    # The settings fixture allows 3 rounds; this model asks for a tool forever.
    server.queue(
        *(tool_reply((f"call_{i}", "add", '{"a": 1, "b": 1}')) for i in range(3))
    )

    with pytest.raises(MaxIterationsExceededError) as exc_info:
        await make_client().chat("Loop forever")

    assert len(server.requests) == 3  # stopped after exactly 3 requests
    error = exc_info.value
    assert error.iterations == 3
    # The partial conversation is attached: user + 3 x (assistant, tool).
    assert len(error.messages) == 7
    assert "LIGHTSPEED_MAX_TOOL_ITERATIONS" in str(error)


# --------------------------------------------------------------------------- #
# 6. API failures become our own exceptions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("reply", "expected", "message_part"),
    [
        (error_reply(429, "slow down"), LLMRateLimitError, "slow down"),
        (error_reply(401, "bad key"), LLMAuthenticationError, "bad key"),
        (error_reply(403, "not allowed"), LLMAuthenticationError, "not allowed"),
        (error_reply(404, "model 'x' not found"), LLMResponseError, "HTTP 404"),
        (error_reply(500, "boom"), LLMResponseError, "boom"),
        (httpx2.ConnectError("refused"), LLMConnectionError, "http://fake.test/v1"),
        (httpx2.ReadTimeout("slow"), LLMTimeoutError, "did not answer in time"),
    ],
    ids=["429", "401", "403", "404", "500", "connect", "timeout"],
)
async def test_api_failures(
    server: FakeLLMServer,
    make_client: ClientFactory,
    reply: Reply,
    expected: type[LLMError],
    message_part: str,
) -> None:
    server.queue(reply)

    with pytest.raises(expected) as exc_info:
        await make_client().chat("Hi")

    assert message_part in str(exc_info.value)
    # ``raise ... from exc`` keeps the SDK's original exception reachable.
    assert isinstance(exc_info.value.__cause__, openai.OpenAIError)


async def test_provider_message_is_kept(
    server: FakeLLMServer, make_client: ClientFactory
) -> None:
    server.queue(error_reply(404, "model 'x' not found"))

    with pytest.raises(LLMResponseError) as exc_info:
        await make_client().chat("Hi")

    assert exc_info.value.provider_message == "model 'x' not found"


async def test_failure_during_tool_loop(
    server: FakeLLMServer, make_client: ClientFactory
) -> None:
    server.queue(
        tool_reply(("call_1", "add", '{"a": 1, "b": 1}')),
        httpx2.ConnectError("server went away"),
    )

    with pytest.raises(LLMConnectionError):
        await make_client().chat("1 + 1?")


def test_unexpected_sdk_error_is_wrapped() -> None:
    # A plain OpenAIError (no HTTP status) falls through to the last branch.
    error = _translate_openai_error(openai.OpenAIError("weird"), "http://x")

    assert isinstance(error, LLMResponseError)
    assert "weird" in str(error)


# --------------------------------------------------------------------------- #
# Unusual responses
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("message", "expected_reply"),
    [
        (
            {"content": None, "refusal": "I cannot help with that."},
            "I cannot help with that.",
        ),
        ({"content": None}, ""),
    ],
    ids=["refusal", "empty"],
)
async def test_reply_without_content(
    server: FakeLLMServer,
    make_client: ClientFactory,
    message: dict[str, Any],
    expected_reply: str,
) -> None:
    server.queue(completion(message))

    result = await make_client().chat("Hi")

    assert result.reply == expected_reply


async def test_cut_off_answer_and_missing_usage(
    server: FakeLLMServer, make_client: ClientFactory
) -> None:
    server.queue(text_reply("This answer is cut o", finish_reason="length", usage=None))

    result = await make_client().chat("Tell me a long story")

    assert result.finish_reason == "length"
    assert result.usage == TokenUsage()  # server did not report usage


async def test_no_choices(server: FakeLLMServer, make_client: ClientFactory) -> None:
    empty = httpx2.Response(
        200,
        json={
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [],
        },
    )
    server.queue(empty)

    with pytest.raises(LLMResponseError, match="empty answer"):
        await make_client().chat("Hi")


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #
async def test_tool_events(server: FakeLLMServer, make_client: ClientFactory) -> None:
    server.queue(tool_reply(("call_1", "add", '{"a": 1, "b": 2}')), text_reply("3"))
    events: list[ToolEvent] = []

    # events.append is a function taking one argument: a perfect callback.
    await make_client(on_event=events.append).chat("1 + 2?")

    started, finished = events
    assert started == ToolStarted("call_1", "add", {"a": 1, "b": 2})
    assert isinstance(finished, ToolFinished)
    assert finished.result.output == 3
    assert finished.duration_seconds >= 0


# --------------------------------------------------------------------------- #
# Construction and configuration
# --------------------------------------------------------------------------- #
def test_missing_api_key_for_openai_is_a_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        AssistantClient(Settings())


def test_explicit_openai_url_still_needs_a_key() -> None:
    settings = Settings(openai_base_url=HttpUrl("https://api.openai.com/v1"))

    with pytest.raises(ConfigurationError):
        AssistantClient(settings)


def test_local_server_gets_placeholder_key() -> None:
    settings = Settings(openai_base_url=HttpUrl("http://localhost:11434/v1"))

    client = AssistantClient(settings)

    # Tests may peek at "private" attributes (leading underscore) to check
    # internal wiring that has no public accessor.
    assert client._openai.api_key == PLACEHOLDER_API_KEY
    assert str(client._openai.base_url) == "http://localhost:11434/v1/"


def test_settings_are_passed_to_the_sdk() -> None:
    settings = Settings(
        openai_api_key=SecretStr("sk-real"),
        request_timeout=12.5,
        llm_max_retries=4,
    )

    client = AssistantClient(settings)

    assert client._openai.api_key == "sk-real"
    assert client._openai.timeout == 12.5
    assert client._openai.max_retries == 4
    assert client._registry is default_registry


async def test_async_with_closes_its_own_client() -> None:
    settings = Settings(openai_api_key=SecretStr("k"))

    async with AssistantClient(settings) as client:
        assert not client._openai.is_closed()

    assert client._openai.is_closed()


async def test_injected_client_is_left_open(make_client: ClientFactory) -> None:
    client = make_client()

    await client.aclose()

    # Whoever created a resource closes it; we did not create this one.
    assert not client._openai.is_closed()


def test_tool_call_round_trip_through_wire_format() -> None:
    history = [
        ChatMessage.system("rules"),
        ChatMessage.assistant(
            tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 1})]
        ),
        ChatMessage.tool("c1", "1"),
    ]

    assert _to_openai_messages(history) == [
        {"role": "system", "content": "rules"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "add", "arguments": '{"a": 1}'},
                }
            ],
        },
        {"role": "tool", "content": "1", "tool_call_id": "c1"},
    ]


async def test_error_body_without_message_field(
    server: FakeLLMServer, make_client: ClientFactory
) -> None:
    # FastAPI-based servers (e.g. vLLM) send {"detail": ...} instead of
    # {"error": {"message": ...}}; we fall back to the SDK's own text.
    server.queue(httpx2.Response(404, json={"detail": "Not Found"}))

    with pytest.raises(LLMResponseError) as exc_info:
        await make_client().chat("Hi")

    assert "Not Found" in str(exc_info.value)


# --------------------------------------------------------------------------- #
# Model and temperature overrides (used by the CLI's --model/--temperature)
# --------------------------------------------------------------------------- #
async def test_model_and_temperature_overrides(
    server: FakeLLMServer, make_client: ClientFactory
) -> None:
    server.queue(text_reply("ok"))
    client = make_client(model="llama3.1", temperature=0.2)

    await client.chat("Hi")

    assert client.model == "llama3.1"
    assert server.requests[0]["model"] == "llama3.1"
    assert server.requests[0]["temperature"] == 0.2


async def test_temperature_is_omitted_by_default(
    server: FakeLLMServer, make_client: ClientFactory
) -> None:
    server.queue(text_reply("ok"))
    client = make_client()

    await client.chat("Hi")

    assert client.model == "test-model"  # from settings
    assert client.base_url == "http://fake.test/v1/"
    # Some models reject "temperature" entirely, so it is not sent unless set.
    assert "temperature" not in server.requests[0]


# --------------------------------------------------------------------------- #
# trim_history
# --------------------------------------------------------------------------- #
def turn(number: int, *, with_tool: bool = False) -> list[ChatMessage]:
    """One question/answer turn, optionally with a tool call in between."""
    messages = [ChatMessage.user(f"question {number}")]
    if with_tool:
        call = ToolCall(id=f"call_{number}", name="add")
        messages += [
            ChatMessage.assistant(tool_calls=[call]),
            ChatMessage.tool(f"call_{number}", "2"),
        ]
    messages.append(ChatMessage.assistant(f"answer {number}"))
    return messages


def test_trim_history_keeps_short_conversations() -> None:
    history = [*turn(1), *turn(2)]

    assert trim_history(history, max_turns=2) == tuple(history)


def test_trim_history_drops_oldest_whole_turns() -> None:
    history = [*turn(1), *turn(2, with_tool=True), *turn(3)]

    trimmed = trim_history(history, max_turns=2)

    # Turn 1 is gone; turn 2 starts the history, with its tool call AND its
    # result still together.
    assert trimmed[0] == ChatMessage.user("question 2")
    assert [m.role for m in trimmed] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
        "assistant",
    ]
