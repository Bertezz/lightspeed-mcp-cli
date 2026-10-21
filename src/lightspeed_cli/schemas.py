"""Core domain schemas: chat messages, tool contracts and execution results.

Learning notes
--------------
A *schema* describes what a piece of data must look like. Each class below is a
Pydantic ``BaseModel``: you declare fields with type hints, and Pydantic

* checks every object you create (wrong data raises ``ValidationError``),
* converts objects to JSON (``model_dump_json()``) and back
  (``model_validate_json()``). JSON is the text format APIs use to exchange data.

These models are the shared vocabulary of the whole program: the AI client, the
tool registry and the CLI all pass these objects around.

A conversation with an AI is a list of ``ChatMessage`` objects. When the AI
wants a tool run, its message contains ``ToolCall`` objects; we run the tool,
get a ``ToolExecutionResult``, and send it back as a ``tool`` message.

All models are immutable. ``dict`` fields cannot be reassigned but their
contents can still be mutated; treat them as read-only.
"""

import json
from collections.abc import Sequence
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

# Who can speak in a conversation:
#   system    -> instructions from us to the AI ("You are a Linux assistant")
#   user      -> the human
#   assistant -> the AI
#   tool      -> our program, reporting the result of a tool the AI asked for
Role = Literal["system", "user", "assistant", "tool"]
ToolStatus = Literal["success", "error"]

# A *regular expression* (pattern for text). It reads as: from start (^) to end
# ($), 1 to 64 characters, each a letter, digit, "_" or "-". OpenAI rejects tool
# names that do not match, so we reject them first, with a clearer error.
TOOL_NAME_PATTERN = r"^[a-zA-Z0-9_-]{1,64}$"


class _FrozenModel(BaseModel):
    """Shared base class: every model below inherits these two rules."""

    # frozen=True    -> objects cannot be changed after creation. A message in a
    #                   conversation history should never be silently edited.
    # extra="forbid" -> unknown fields are errors, so a typo like contnet="hi"
    #                   fails loudly instead of being silently ignored.
    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
class ToolCall(_FrozenModel):
    """A tool invocation requested by the assistant."""

    # The id links a request to its answer: our result message repeats it in
    # ``tool_call_id`` so the AI knows which request the result belongs to.
    id: str = Field(min_length=1)
    name: str = Field(pattern=TOOL_NAME_PATTERN)
    # ``JsonValue`` = "anything JSON can represent" (str, int, float, bool, None,
    # lists and dicts of those). Safer than ``Any``, which disables type checks.
    # ``default_factory=dict`` creates a *new* empty dict for every object;
    # a plain ``= {}`` default would be one dict shared by all objects.
    arguments: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("arguments", mode="before")
    @classmethod
    def _parse_json_arguments(cls, value: object) -> object:
        """Accept arguments as a JSON-encoded string, as the OpenAI API sends."""
        # OpenAI sends arguments as *text containing JSON*: '{"interval": 1}'.
        # This turns that text into a real dict before normal validation runs.
        if not isinstance(value, str):
            return value  # already a dict (or junk Pydantic will reject)
        if not value.strip():
            return {}  # "" means "no arguments"
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            # ``raise ... from exc`` keeps the original error attached, which
            # helps when debugging. Pydantic turns ValueError into a
            # ValidationError with this message.
            raise ValueError(f"arguments is not valid JSON: {exc.msg}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("arguments JSON must be an object")
        return parsed


class ChatMessage(_FrozenModel):
    """A single message in a chat conversation."""

    role: Role
    content: str | None = None
    # A tuple is an unchangeable list, which fits a frozen model.
    tool_calls: tuple[ToolCall, ...] | None = None
    tool_call_id: str | None = None

    # A *model* validator sees all fields at once, so it can check rules that
    # involve several fields. ``mode="after"`` runs once the fields are filled.
    @model_validator(mode="after")
    def _check_role_fields(self) -> Self:
        if self.tool_calls is not None and self.role != "assistant":
            raise ValueError("tool_calls is only allowed on assistant messages")
        if self.role == "tool":
            if not self.tool_call_id:
                raise ValueError("tool messages require tool_call_id")
        elif self.tool_call_id is not None:
            raise ValueError("tool_call_id is only allowed on tool messages")
        # Only an assistant message that asks for tools may have no text.
        if self.content is None and not (self.role == "assistant" and self.tool_calls):
            raise ValueError(
                "content is required unless an assistant message has tool_calls"
            )
        return self

    # Convenience constructors: ChatMessage.user("hi") instead of
    # ChatMessage(role="user", content="hi"). ``@classmethod`` means the method
    # is called on the class, and receives the class as ``cls``. ``Self`` means
    # "an instance of this class".
    @classmethod
    def system(cls, content: str) -> Self:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> Self:
        return cls(role="user", content=content)

    @classmethod
    def assistant(
        cls,
        content: str | None = None,
        tool_calls: Sequence[ToolCall] | None = None,
    ) -> Self:
        # Accept any sequence (e.g. a list) for convenience; store a tuple.
        return cls(
            role="assistant",
            content=content,
            tool_calls=tuple(tool_calls) if tool_calls is not None else None,
        )

    @classmethod
    def tool(cls, tool_call_id: str, content: str) -> Self:
        return cls(role="tool", content=content, tool_call_id=tool_call_id)


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
class ToolParameterSchema(_FrozenModel):
    """Top-level JSON Schema describing a tool's arguments.

    Individual property schemas are kept as raw JSON, and unknown top-level
    keywords (e.g. ``$defs``) pass through, so MCP schemas round-trip intact.

    Example (a tool taking one required number called ``interval``)::

        {
            "type": "object",
            "properties": {"interval": {"type": "number"}},
            "required": ["interval"],
            "additionalProperties": false,
        }
    """

    # This model_config is *merged* with the parent's: it stays frozen, but
    # allows extra keys (so JSON Schema keywords we do not model survive).
    # JSON Schema spells it "additionalProperties" (camelCase) while Python style
    # is snake_case; the alias settings accept both and always output camelCase.
    model_config = ConfigDict(
        extra="allow",
        validate_by_name=True,
        validate_by_alias=True,
        serialize_by_alias=True,
    )

    type: Literal["object"] = "object"
    properties: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)
    required: tuple[str, ...] = ()
    additional_properties: bool = Field(default=False, alias="additionalProperties")

    @model_validator(mode="after")
    def _check_required(self) -> Self:
        # Saying "b is required" when there is no property b is a mistake.
        missing = [name for name in self.required if name not in self.properties]
        if missing:
            raise ValueError(f"required names not in properties: {', '.join(missing)}")
        return self


class ToolDefinition(_FrozenModel):
    """Metadata describing a tool that the assistant may call."""

    name: str = Field(pattern=TOOL_NAME_PATTERN)
    # The AI reads the description to decide *when* to use the tool, so it is
    # required and should be written for the AI, in plain language.
    description: str = Field(min_length=1)
    parameters: ToolParameterSchema = Field(default_factory=ToolParameterSchema)

    def to_openai_tool(self) -> dict[str, JsonValue]:
        """Return the definition in OpenAI ``tools`` request format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters.model_dump(mode="json", by_alias=True),
            },
        }

    def to_mcp_tool(self) -> dict[str, JsonValue]:
        """Return the definition in MCP ``tools/list`` format.

        Same information as :meth:`to_openai_tool`, different layout: MCP calls
        the schema ``inputSchema`` and does not nest it under ``"function"``.
        """
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.parameters.model_dump(mode="json", by_alias=True),
        }


class ToolExecutionResult(_FrozenModel):
    """The outcome of executing a tool."""

    status: ToolStatus
    output: JsonValue = None
    error: str | None = None

    @model_validator(mode="after")
    def _check_status(self) -> Self:
        if self.status == "error" and not self.error:
            raise ValueError("error results require an error message")
        if self.status == "success" and self.error is not None:
            raise ValueError("success results must not set error")
        return self

    # Shortcuts: ToolExecutionResult.ok(42) / ToolExecutionResult.fail("boom").
    @classmethod
    def ok(cls, output: JsonValue = None) -> Self:
        return cls(status="success", output=output)

    @classmethod
    def fail(cls, error: str) -> Self:
        return cls(status="error", error=error)

    def to_message(self, tool_call_id: str) -> ChatMessage:
        """Build the ``tool`` message that reports this result to the model."""
        # Message content must be text, so structured output is JSON-encoded.
        # ensure_ascii=False keeps characters like "é" readable instead of
        # escaping them as "é".
        if self.status == "error":
            content = f"Error: {self.error}"
        elif isinstance(self.output, str):
            content = self.output
        else:
            content = json.dumps(self.output, ensure_ascii=False)
        return ChatMessage.tool(tool_call_id, content)


# --------------------------------------------------------------------------- #
# Results of a chat turn (produced by client.AssistantClient)
# --------------------------------------------------------------------------- #
class TokenUsage(_FrozenModel):
    """How many tokens (pieces of words) the model read and wrote.

    Providers bill per token, so this is worth showing. Some local servers do
    not report usage; then everything stays 0.
    """

    prompt_tokens: int = 0  # tokens sent to the model
    completion_tokens: int = 0  # tokens the model generated
    total_tokens: int = 0

    def __add__(self, other: Self) -> Self:
        # Defining __add__ makes ``usage_a + usage_b`` work. One question can
        # take several requests, and we add up the usage of all of them.
        return type(self)(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class ToolRun(_FrozenModel):
    """One tool execution that happened while answering a question."""

    call_id: str
    name: str
    arguments: dict[str, JsonValue]
    result: ToolExecutionResult
    duration_seconds: float = Field(ge=0)


class ChatResult(_FrozenModel):
    """Everything produced by one ``AssistantClient.chat()`` call."""

    reply: str
    """The assistant's final answer."""

    messages: tuple[ChatMessage, ...]
    """The whole conversation so far (without the system prompt). Pass it as
    ``history`` to the next ``chat()`` call to continue the conversation."""

    usage: TokenUsage = Field(default_factory=TokenUsage)
    """Tokens used, added up over every request made for this question."""

    iterations: int = Field(ge=1)
    """How many requests were sent to the model for this question."""

    tool_runs: tuple[ToolRun, ...] = ()
    """Every tool executed along the way, in order."""

    finish_reason: str | None = None
    """Why the model stopped: "stop" is normal; "length" means the answer was
    cut off because it hit the maximum output length."""
