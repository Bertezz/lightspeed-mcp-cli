"""Core domain schemas: chat messages, tool contracts and execution results.

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

Role = Literal["system", "user", "assistant", "tool"]
ToolStatus = Literal["success", "error"]

# OpenAI function-name constraint.
TOOL_NAME_PATTERN = r"^[a-zA-Z0-9_-]{1,64}$"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
class ToolCall(_FrozenModel):
    """A tool invocation requested by the assistant."""

    id: str = Field(min_length=1)
    name: str = Field(pattern=TOOL_NAME_PATTERN)
    arguments: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("arguments", mode="before")
    @classmethod
    def _parse_json_arguments(cls, value: object) -> object:
        """Accept arguments as a JSON-encoded string, as the OpenAI API sends."""
        if not isinstance(value, str):
            return value
        if not value.strip():
            return {}
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"arguments is not valid JSON: {exc.msg}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("arguments JSON must be an object")
        return parsed


class ChatMessage(_FrozenModel):
    """A single message in a chat conversation."""

    role: Role
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] | None = None
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def _check_role_fields(self) -> Self:
        if self.tool_calls is not None and self.role != "assistant":
            raise ValueError("tool_calls is only allowed on assistant messages")
        if self.role == "tool":
            if not self.tool_call_id:
                raise ValueError("tool messages require tool_call_id")
        elif self.tool_call_id is not None:
            raise ValueError("tool_call_id is only allowed on tool messages")
        if self.content is None and not (self.role == "assistant" and self.tool_calls):
            raise ValueError(
                "content is required unless an assistant message has tool_calls"
            )
        return self

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
    """

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
        missing = [name for name in self.required if name not in self.properties]
        if missing:
            raise ValueError(f"required names not in properties: {', '.join(missing)}")
        return self


class ToolDefinition(_FrozenModel):
    """Metadata describing a tool that the assistant may call."""

    name: str = Field(pattern=TOOL_NAME_PATTERN)
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

    @classmethod
    def ok(cls, output: JsonValue = None) -> Self:
        return cls(status="success", output=output)

    @classmethod
    def fail(cls, error: str) -> Self:
        return cls(status="error", error=error)

    def to_message(self, tool_call_id: str) -> ChatMessage:
        """Build the ``tool`` message that reports this result to the model."""
        if self.status == "error":
            content = f"Error: {self.error}"
        elif isinstance(self.output, str):
            content = self.output
        else:
            content = json.dumps(self.output, ensure_ascii=False)
        return ChatMessage.tool(tool_call_id, content)
