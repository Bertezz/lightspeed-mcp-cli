"""The assistant: talks to the model server and runs tools when it asks.

Learning notes
--------------
**The agent loop.** A language model cannot run code. It can only *reply*, but
the reply may be a request: "please call get_system_metrics". So answering one
question can take several rounds::

    you  -> model:  "How busy is my CPU?"          (+ the list of tools)
    model -> you:   call get_system_metrics()       (a tool call, no text)
    you:            run the tool                    (registry.execute)
    you  -> model:  everything so far + the result
    model -> you:   "Your CPU is at 12%."           (text: we are done)

``AssistantClient.chat`` runs that loop. Each round is a full HTTP request
containing the *whole* conversation, because the model remembers nothing
between requests. A safety limit (``max_tool_iterations``) stops a model that
keeps asking for tools and never answers.

**async/await.** Waiting for the model server can take seconds. ``async``
functions let the program do other work meanwhile; here, running several tool
calls from the same response at the same time with ``asyncio.gather``.

**Stateless design.** The client keeps no conversation. ``chat()`` takes the
history in and returns the new history in ``ChatResult.messages``; the caller
decides what to keep. One client can therefore serve many conversations, and
what gets sent is always visible.

**One adapter.** This is the only module that knows the OpenAI message format
and the ``openai`` SDK's exception classes. The rest of the program speaks our
own models (``ChatMessage``, ``ToolCall``, ...) and our own exceptions.

Works with any OpenAI-compatible server: OpenAI, Groq, Ollama, vLLM, ...
"""

import asyncio
import json
import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Self, cast

import openai
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallUnion,
    ChatCompletionToolUnionParam,
)
from pydantic import JsonValue, ValidationError

from lightspeed_cli.config import Settings, get_settings
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
from lightspeed_cli.schemas import (
    TOOL_NAME_PATTERN,
    ChatMessage,
    ChatResult,
    TokenUsage,
    ToolCall,
    ToolExecutionResult,
    ToolRun,
)
from lightspeed_cli.tools import ToolRegistry
from lightspeed_cli.tools import registry as default_registry

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are Lightspeed, a command-line assistant for Linux users and software "
    "developers. Use the available tools to look up facts about the user's "
    "machine and Git repositories instead of guessing. If a tool returns an "
    "error, explain it plainly or try again with corrected arguments. Keep "
    "answers concise and practical."
)

OPENAI_HOST = "api.openai.com"

# The openai SDK refuses to start without *some* API key, but local servers
# such as Ollama and vLLM ignore it. This placeholder satisfies the SDK.
PLACEHOLDER_API_KEY = "not-needed"

# Stands in for a tool name the model invented that our schemas cannot store.
INVALID_TOOL_NAME = "invalid_tool_name"

# How many question/answer turns trim_history keeps (see its docstring).
MAX_HISTORY_TURNS = 20


# --------------------------------------------------------------------------- #
# Events: how the client reports progress without printing anything itself
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ToolStarted:
    call_id: str
    name: str
    arguments: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class ToolFinished:
    call_id: str
    name: str
    result: ToolExecutionResult
    duration_seconds: float


# A *union* type alias: an event is one of these two classes. A callback can
# use ``isinstance(event, ToolStarted)`` to tell them apart.
ToolEvent = ToolStarted | ToolFinished


class AssistantClient:
    """Chat with an OpenAI-compatible model that can use our registered tools.

    Example::

        async with AssistantClient() as client:
            result = await client.chat("How busy is my CPU?")
            print(result.reply)
            follow_up = await client.chat("And memory?", result.messages)
    """

    def __init__(
        self,
        settings: Settings | None = None,
        registry: ToolRegistry | None = None,
        *,
        model: str | None = None,
        temperature: float | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        on_event: Callable[[ToolEvent], None] | None = None,
        openai_client: AsyncOpenAI | None = None,
    ) -> None:
        """Create a client.

        Args:
            settings: Configuration; defaults to ``get_settings()``.
            registry: Tools the model may call; defaults to the built-in tools.
            model: Model name; defaults to ``settings.default_model``.
            temperature: Randomness of the answers (0 = most predictable). When
                ``None``, nothing is sent and the server's default applies;
                some models reject the field entirely.
            system_prompt: Instructions sent before every conversation.
            on_event: Called when a tool starts and finishes (e.g. to show
                progress in the terminal).
            openai_client: A ready-made SDK client, mainly for tests. A client
                passed in here is *not* closed by ``aclose()``: whoever
                creates a resource is responsible for closing it.

        Raises:
            ConfigurationError: no API key is set but OpenAI's servers are used.
        """
        self._settings = settings or get_settings()
        # ``registry or default_registry`` would be a bug: an *empty* registry
        # has len() == 0, which Python treats as False, so it would silently be
        # replaced by the default one. ``is not None`` checks only for "missing".
        self._registry = registry if registry is not None else default_registry
        self._model = model or self._settings.default_model
        self._temperature = temperature
        self._system_prompt = system_prompt
        self._on_event = on_event
        self._owns_openai_client = openai_client is None
        self._openai = openai_client or AsyncOpenAI(
            api_key=_resolve_api_key(self._settings),
            base_url=self._settings.base_url_str(),  # None -> OpenAI's servers
            timeout=self._settings.request_timeout,
            max_retries=self._settings.llm_max_retries,
        )

    # Read-only properties: callers can see these values but not change them.
    @property
    def model(self) -> str:
        """The model this client sends requests to."""
        return self._model

    @property
    def base_url(self) -> str:
        """The server this client talks to."""
        return str(self._openai.base_url)

    # ----------------------------------------------------------------------- #
    # Closing. ``async with`` calls __aenter__ on entry and __aexit__ on exit,
    # even when an exception is raised inside the block.
    # ----------------------------------------------------------------------- #
    async def aclose(self) -> None:
        """Close the network connections, if this client created them."""
        if self._owns_openai_client:
            await self._openai.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    # ----------------------------------------------------------------------- #
    # The agent loop
    # ----------------------------------------------------------------------- #
    async def chat(
        self, prompt: str, history: Sequence[ChatMessage] = ()
    ) -> ChatResult:
        """Answer ``prompt``, calling tools as often as the model asks.

        Args:
            prompt: The user's new message.
            history: Earlier messages (``ChatResult.messages`` of the previous
                turn) to continue a conversation.

        Raises:
            LLMError: talking to the model server failed (subclasses say why).
            MaxIterationsExceededError: the model never stopped calling tools.
        """
        messages: list[ChatMessage] = [*history, ChatMessage.user(prompt)]
        usage = TokenUsage()
        tool_runs: list[ToolRun] = []
        limit = self._settings.max_tool_iterations

        for iteration in range(1, limit + 1):
            response = await self._complete(messages)
            usage = usage + _usage_of(response)
            if not response.choices:
                raise LLMResponseError("The model server returned an empty answer.")
            choice = response.choices[0]
            message = choice.message

            # No tool calls: this is the final answer.
            if not message.tool_calls:
                # A model may decline to answer; the reason is in "refusal".
                reply = message.content or message.refusal or ""
                messages.append(ChatMessage.assistant(reply))
                return ChatResult(
                    reply=reply,
                    messages=tuple(messages),
                    usage=usage,
                    iterations=iteration,
                    tool_runs=tuple(tool_runs),
                    finish_reason=choice.finish_reason,
                )

            # Tool calls: record the request, run the tools, and go round again.
            # The assistant message must come *before* the tool results, and
            # every result must carry the id of the call it answers.
            calls = [
                _from_openai_tool_call(raw, iteration, index)
                for index, raw in enumerate(message.tool_calls)
            ]
            messages.append(
                ChatMessage.assistant(
                    message.content, tool_calls=[call for call, _ in calls]
                )
            )
            # gather() starts all the coroutines and waits until every one has
            # finished. The calls run at the same time, and the results come
            # back in the same order as the calls.
            runs = await asyncio.gather(
                *(self._run_tool(call, problem) for call, problem in calls)
            )
            tool_runs.extend(runs)
            messages.extend(run.result.to_message(run.call_id) for run in runs)

        raise MaxIterationsExceededError(iterations=limit, messages=tuple(messages))

    async def _complete(self, messages: Sequence[ChatMessage]) -> ChatCompletion:
        """Send one request to the model server."""
        tools = self._registry.get_openai_tools()
        logger.debug(
            "Requesting a completion: %d messages, %d tools", len(messages), len(tools)
        )
        try:
            return await self._openai.chat.completions.create(
                model=self._model,
                temperature=(
                    self._temperature if self._temperature is not None else openai.omit
                ),
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    *_to_openai_messages(messages),
                ],
                # ``cast`` tells mypy "trust me, this has that type"; it does
                # nothing at runtime. Safe here because ToolDefinition builds
                # exactly the OpenAI tool format (and a test checks it).
                # ``openai.omit`` leaves the field out of the request entirely:
                # some servers reject an empty "tools": [] list.
                tools=cast("list[ChatCompletionToolUnionParam]", tools)
                if tools
                else openai.omit,
            )
        except openai.OpenAIError as exc:
            raise _translate_openai_error(exc, str(self._openai.base_url)) from exc

    async def _run_tool(self, call: ToolCall, problem: str | None) -> ToolRun:
        """Run one tool call (or report why it cannot run) and time it."""
        self._emit(ToolStarted(call.id, call.name, call.arguments))
        started = time.perf_counter()  # a clock meant for measuring durations
        if problem is not None:
            result = ToolExecutionResult.fail(problem)
        else:
            # execute() never raises for tool problems: a broken tool becomes
            # an error result the model can read, and the chat carries on.
            result = await self._registry.execute(call.name, call.arguments)
        duration = time.perf_counter() - started
        # DEBUG, not INFO: the CLI already shows a line per tool, so at the
        # default LOG_LEVEL=INFO this would print everything twice.
        logger.debug("Tool %s finished: %s (%.2fs)", call.name, result.status, duration)
        self._emit(ToolFinished(call.id, call.name, result, duration))
        return ToolRun(
            call_id=call.id,
            name=call.name,
            arguments=call.arguments,
            result=result,
            duration_seconds=duration,
        )

    def _emit(self, event: ToolEvent) -> None:
        if self._on_event is not None:
            self._on_event(event)


def trim_history(
    messages: Sequence[ChatMessage], max_turns: int = MAX_HISTORY_TURNS
) -> tuple[ChatMessage, ...]:
    """Keep only the last ``max_turns`` turns of a conversation.

    Every model has a *context window*: a maximum amount of text per request.
    Since each request contains the whole conversation, a long chat would
    eventually be rejected. We therefore drop the oldest turns.

    A turn starts at a user message and includes everything after it (tool
    calls, tool results, the answer). Cutting only at user messages keeps each
    tool call together with its result; a result without its call would make
    the server reject the request.
    """
    turn_starts = [i for i, message in enumerate(messages) if message.role == "user"]
    if len(turn_starts) <= max_turns:
        return tuple(messages)
    return tuple(messages[turn_starts[-max_turns] :])


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def _resolve_api_key(settings: Settings) -> str:
    """Pick the API key to give the SDK (see PLACEHOLDER_API_KEY)."""
    if settings.openai_api_key is not None:
        return settings.openai_api_key.get_secret_value()
    host = settings.openai_base_url.host if settings.openai_base_url else OPENAI_HOST
    if host == OPENAI_HOST:
        raise ConfigurationError(
            "OPENAI_API_KEY is not set, but OpenAI's servers need one. Set it in "
            "your .env file, or point OPENAI_BASE_URL at a local server such as "
            "Ollama (http://localhost:11434/v1)."
        )
    # Other servers decide for themselves: local ones ignore the key, hosted ones
    # (e.g. Groq) reply 401, which becomes a clear LLMAuthenticationError.
    return PLACEHOLDER_API_KEY


# --------------------------------------------------------------------------- #
# Converting between our models and the OpenAI wire format
# --------------------------------------------------------------------------- #
def _to_openai_messages(
    messages: Sequence[ChatMessage],
) -> list[ChatCompletionMessageParam]:
    """Turn our ChatMessage objects into the dicts the SDK sends as JSON."""
    converted: list[ChatCompletionMessageParam] = []
    for message in messages:
        if message.role == "system":
            converted.append({"role": "system", "content": message.content or ""})
        elif message.role == "user":
            converted.append({"role": "user", "content": message.content or ""})
        elif message.role == "assistant":
            assistant: ChatCompletionAssistantMessageParam = {
                "role": "assistant",
                "content": message.content,
            }
            if message.tool_calls:
                assistant["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        # The API expects arguments as a JSON *string*.
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in message.tool_calls
                ]
            converted.append(assistant)
        else:
            converted.append(
                {
                    "role": "tool",
                    "content": message.content or "",
                    "tool_call_id": message.tool_call_id or "",
                }
            )
    return converted


def _from_openai_tool_call(
    raw: ChatCompletionMessageToolCallUnion, iteration: int, index: int
) -> tuple[ToolCall, str | None]:
    """Convert one tool call from the SDK into our ToolCall.

    Returns the call plus a *problem* description when it cannot be run. Then
    the tool is not executed; the model gets the problem as an error result and
    can try again. Raising instead would end the whole conversation because of
    one malformed request.
    """
    # Some local servers leave the id empty; invent a unique one.
    call_id = raw.id or f"call_{iteration}_{index}"

    if not isinstance(raw, ChatCompletionMessageFunctionToolCall):
        # A "custom" tool call: we only offer function tools, so we cannot run it.
        name = raw.custom.name
        call = ToolCall(id=call_id, name=_storable_name(name), arguments={})
        return call, f"Unsupported tool call type {raw.type!r}; call a function tool"

    name = raw.function.name
    if _storable_name(name) != name:
        call = ToolCall(id=call_id, name=INVALID_TOOL_NAME, arguments={})
        return call, f"{name!r} is not a valid tool name"

    try:
        # ToolCall's validator parses the JSON string of arguments for us.
        call = ToolCall.model_validate(
            {"id": call_id, "name": name, "arguments": raw.function.arguments}
        )
    except ValidationError:
        # Keep the call in the history (with no arguments) so the model sees
        # which request failed; quote the start of the broken text in the error.
        snippet = raw.function.arguments[:200]
        call = ToolCall(id=call_id, name=name, arguments={})
        return call, (
            f"The arguments were not a valid JSON object: {snippet!r}. "
            "Call the tool again with valid JSON."
        )
    return call, None


def _storable_name(name: str) -> str:
    """Return ``name`` if our schemas accept it, else a placeholder."""
    return name if re.fullmatch(TOOL_NAME_PATTERN, name) else INVALID_TOOL_NAME


def _usage_of(response: ChatCompletion) -> TokenUsage:
    if response.usage is None:  # some local servers do not report usage
        return TokenUsage()
    return TokenUsage(
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        total_tokens=response.usage.total_tokens,
    )


# --------------------------------------------------------------------------- #
# Translating SDK exceptions into ours
# --------------------------------------------------------------------------- #
def _translate_openai_error(exc: openai.OpenAIError, base_url: str) -> LLMError:
    """Map an ``openai`` exception to the matching LLMError subclass.

    The ``isinstance`` checks go from most to least specific. Order matters:
    APITimeoutError is a *subclass* of APIConnectionError, so checking for
    APIConnectionError first would report every timeout as a connection error.
    """
    detail = _provider_detail(exc)
    if isinstance(exc, openai.APITimeoutError):
        return LLMTimeoutError(
            f"The model server at {base_url} did not answer in time. It may be "
            "overloaded, or the model may still be loading; try again or raise "
            "REQUEST_TIMEOUT."
        )
    if isinstance(exc, openai.APIConnectionError):
        return LLMConnectionError(
            f"Could not reach the model server at {base_url}. Check "
            "OPENAI_BASE_URL and that the server is running."
        )
    if isinstance(exc, openai.RateLimitError):
        return LLMRateLimitError(
            "The model provider is rate-limiting requests (HTTP 429), even after "
            f"retrying. Wait a moment and try again. Provider said: {detail}",
            provider_message=detail,
        )
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return LLMAuthenticationError(
            f"The model server rejected the API key (HTTP {exc.status_code}). "
            f"Check OPENAI_API_KEY. Provider said: {detail}",
            provider_message=detail,
        )
    if isinstance(exc, openai.APIStatusError):
        return LLMResponseError(
            f"The model server rejected the request (HTTP {exc.status_code}): {detail}",
            provider_message=detail,
        )
    return LLMResponseError(f"Unexpected error from the model client: {detail}")


def _provider_detail(exc: openai.OpenAIError) -> str:
    """Extract the server's own error message, e.g. "model 'x' not found"."""
    if isinstance(exc, openai.APIError) and isinstance(exc.body, dict):
        message = exc.body.get("message")
        if isinstance(message, str):
            return message
    return str(exc)
