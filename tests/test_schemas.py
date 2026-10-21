"""Tests for the core domain schemas.

Learning notes
--------------
* ``Model.model_validate(dict)`` builds a model from plain Python data. Tests use
  it for *invalid* input, because writing ``ChatMessage(role="robot")`` directly
  would already be flagged by mypy before the test even runs.
* A "round trip" (object -> JSON -> object) proves nothing is lost when data
  travels to the AI API and back.
* ``ids=[...]`` in ``parametrize`` gives each case a readable name in the pytest
  output, e.g. ``test_chat_message_rejects_invalid[tool-without-id]``.
"""

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from lightspeed_cli.schemas import (
    ChatMessage,
    ToolCall,
    ToolDefinition,
    ToolExecutionResult,
    ToolParameterSchema,
)

CPU_CALL = ToolCall(id="call_1", name="get_cpu", arguments={"interval": 1})
CPU_PARAMS = ToolParameterSchema(
    properties={"interval": {"type": "number"}},
    required=("interval",),
)
CPU_TOOL = ToolDefinition(
    name="get_cpu", description="Read CPU usage.", parameters=CPU_PARAMS
)


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "model",
    [
        CPU_CALL,
        ChatMessage.system("You are helpful."),
        ChatMessage.user("hi"),
        ChatMessage.assistant(tool_calls=[CPU_CALL]),
        ChatMessage.tool("call_1", "42%"),
        CPU_PARAMS,
        CPU_TOOL,
        ToolExecutionResult.ok({"cpu": 12.5, "cores": [1, 2]}),
        ToolExecutionResult.fail("boom"),
    ],
    ids=lambda model: type(model).__name__,
)
def test_json_round_trip(model: BaseModel) -> None:
    restored = type(model).model_validate_json(model.model_dump_json())

    assert restored == model


# --------------------------------------------------------------------------- #
# ToolCall
# --------------------------------------------------------------------------- #
def test_tool_call_parses_json_string_arguments() -> None:
    call = ToolCall.model_validate(
        {"id": "c1", "name": "f", "arguments": '{"path": "/tmp", "depth": 2}'}
    )

    assert call.arguments == {"path": "/tmp", "depth": 2}


@pytest.mark.parametrize("raw", ["", "   "])
def test_tool_call_blank_arguments_become_empty(raw: str) -> None:
    call = ToolCall.model_validate({"id": "c1", "name": "f", "arguments": raw})

    assert call.arguments == {}


def test_tool_call_arguments_default_to_empty() -> None:
    assert ToolCall(id="c1", name="f").arguments == {}


@pytest.mark.parametrize(
    "payload",
    [
        {"id": "c1", "name": "f", "arguments": "{not json"},
        {"id": "c1", "name": "f", "arguments": "[1, 2]"},
        {"id": "c1", "name": "f", "arguments": 5},
        {"id": "", "name": "f"},
        {"id": "c1", "name": "bad name"},
    ],
    ids=["invalid-json", "json-array", "non-mapping", "empty-id", "bad-name"],
)
def test_tool_call_rejects_invalid(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ToolCall.model_validate(payload)


# --------------------------------------------------------------------------- #
# ChatMessage
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("role", ["system", "user", "assistant", "tool"])
def test_chat_message_accepts_valid_roles(role: str) -> None:
    payload: dict[str, Any] = {"role": role, "content": "x"}
    if role == "tool":
        payload["tool_call_id"] = "c1"

    assert ChatMessage.model_validate(payload).role == role


def test_chat_message_constructors() -> None:
    assert ChatMessage.system("s") == ChatMessage(role="system", content="s")
    assert ChatMessage.user("u") == ChatMessage(role="user", content="u")
    assert ChatMessage.assistant("a") == ChatMessage(role="assistant", content="a")
    assert ChatMessage.tool("c1", "t") == ChatMessage(
        role="tool", content="t", tool_call_id="c1"
    )


def test_assistant_with_tool_calls_needs_no_content() -> None:
    message = ChatMessage.assistant(tool_calls=[CPU_CALL])

    assert message.content is None
    assert message.tool_calls == (CPU_CALL,)


def test_assistant_tool_calls_parse_openai_style_payload() -> None:
    message = ChatMessage.model_validate(
        {
            "role": "assistant",
            "tool_calls": [{"id": "c1", "name": "f", "arguments": '{"a": 1}'}],
        }
    )

    assert message.tool_calls is not None
    assert message.tool_calls[0].arguments == {"a": 1}


@pytest.mark.parametrize(
    "payload",
    [
        {"role": "robot", "content": "x"},
        {"role": "user", "content": "x", "tool_calls": [CPU_CALL.model_dump()]},
        {"role": "tool", "content": "x"},
        {"role": "tool", "content": "x", "tool_call_id": ""},
        {"role": "user", "content": "x", "tool_call_id": "c1"},
        {"role": "user"},
        {"role": "assistant"},
        {"role": "assistant", "tool_calls": []},
        {"role": "user", "content": "x", "unexpected": 1},
    ],
    ids=[
        "unknown-role",
        "tool-calls-on-user",
        "tool-without-id",
        "tool-with-empty-id",
        "id-on-user",
        "user-without-content",
        "assistant-empty",
        "assistant-empty-tool-calls",
        "extra-field",
    ],
)
def test_chat_message_rejects_invalid(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ChatMessage.model_validate(payload)


# --------------------------------------------------------------------------- #
# ToolParameterSchema / ToolDefinition
# --------------------------------------------------------------------------- #
def test_parameter_schema_defaults() -> None:
    assert ToolParameterSchema().model_dump(mode="json") == {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }


def test_parameter_schema_accepts_field_name_and_alias() -> None:
    by_name = ToolParameterSchema(additional_properties=True)
    by_alias = ToolParameterSchema.model_validate({"additionalProperties": True})

    assert by_name.additional_properties is True
    assert by_alias == by_name


def test_parameter_schema_passes_unknown_keywords_through() -> None:
    raw = {
        "type": "object",
        "properties": {"item": {"$ref": "#/$defs/Item"}},
        "required": ["item"],
        "additionalProperties": False,
        "$defs": {"Item": {"type": "string", "enum": ["a", "b"]}},
    }

    schema = ToolParameterSchema.model_validate(raw)

    assert schema.model_dump(mode="json") == raw


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "array"},
        {"properties": {"a": {"type": "string"}}, "required": ["a", "b"]},
        {"properties": {"a": "not-a-schema"}},
    ],
    ids=["non-object-type", "required-not-in-properties", "property-not-mapping"],
)
def test_parameter_schema_rejects_invalid(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ToolParameterSchema.model_validate(payload)


@pytest.mark.parametrize("name", ["get_cpu", "get-cpu-usage", "A1", "a" * 64])
def test_tool_definition_accepts_valid_names(name: str) -> None:
    assert ToolDefinition(name=name, description="d").name == name


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "", "description": "d"},
        {"name": "a" * 65, "description": "d"},
        {"name": "has space", "description": "d"},
        {"name": "dotted.name", "description": "d"},
        {"name": "ok", "description": ""},
    ],
    ids=["empty-name", "too-long", "space", "dot", "empty-description"],
)
def test_tool_definition_rejects_invalid(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ToolDefinition.model_validate(payload)


def test_to_mcp_tool() -> None:
    assert CPU_TOOL.to_mcp_tool() == {
        "name": "get_cpu",
        "description": "Read CPU usage.",
        "inputSchema": {
            "type": "object",
            "properties": {"interval": {"type": "number"}},
            "required": ["interval"],
            "additionalProperties": False,
        },
    }


def test_to_openai_tool() -> None:
    assert CPU_TOOL.to_openai_tool() == {
        "type": "function",
        "function": {
            "name": "get_cpu",
            "description": "Read CPU usage.",
            "parameters": {
                "type": "object",
                "properties": {"interval": {"type": "number"}},
                "required": ["interval"],
                "additionalProperties": False,
            },
        },
    }


# --------------------------------------------------------------------------- #
# ToolExecutionResult
# --------------------------------------------------------------------------- #
def test_execution_result_constructors() -> None:
    success = ToolExecutionResult.ok("done")
    failure = ToolExecutionResult.fail("boom")

    assert (success.status, success.output, success.error) == ("success", "done", None)
    assert (failure.status, failure.output, failure.error) == ("error", None, "boom")


@pytest.mark.parametrize(
    ("result", "expected_content"),
    [
        (ToolExecutionResult.ok("plain text"), "plain text"),
        (
            ToolExecutionResult.ok({"cpu": 12.5, "név": "ü"}),
            '{"cpu": 12.5, "név": "ü"}',
        ),
        (ToolExecutionResult.ok(), "null"),
        (ToolExecutionResult.fail("disk not found"), "Error: disk not found"),
    ],
    ids=["string", "structured", "empty", "error"],
)
def test_execution_result_to_message(
    result: ToolExecutionResult, expected_content: str
) -> None:
    assert result.to_message("call_1") == ChatMessage.tool("call_1", expected_content)


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "error"},
        {"status": "error", "error": ""},
        {"status": "success", "error": "unexpected"},
        {"status": "done"},
    ],
    ids=["error-without-message", "error-empty-message", "success-with-error", "bad"],
)
def test_execution_result_rejects_invalid(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ToolExecutionResult.model_validate(payload)


# --------------------------------------------------------------------------- #
# Immutability
# --------------------------------------------------------------------------- #
def test_models_are_frozen() -> None:
    message = ChatMessage.user("hi")

    with pytest.raises(ValidationError):
        message.content = "changed"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        CPU_TOOL.name = "other"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        CPU_CALL.arguments = {}  # type: ignore[misc]
