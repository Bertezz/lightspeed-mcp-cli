"""Tests for the tool registry (``lightspeed_cli.tools.base``).

Learning notes
--------------
* Every test gets a brand-new, empty ``ToolRegistry`` from the ``registry``
  fixture below, so tests never see each other's tools.
* Tools are defined *inside* the tests. That keeps each test self-contained:
  everything it depends on is visible in one place.
* ``async def`` tests are run by pytest-asyncio (``asyncio_mode = "auto"`` in
  pyproject.toml), so they can ``await registry.execute(...)`` directly.
* Some lines end in ``# type: ignore[...]``. Those deliberately write code that
  mypy would reject (e.g. a parameter without a type hint) to check that the
  registry rejects it too at runtime.
"""

import asyncio
import threading
import time
from typing import Annotated, Any, Literal

import pytest
from pydantic import BaseModel, Field

from lightspeed_cli.schemas import ToolCall, ToolExecutionResult
from lightspeed_cli.tools import DEFAULT_TIMEOUT, ToolRegistry
from lightspeed_cli.tools.base import _strip_titles


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry()


@pytest.fixture
def calculator(registry: ToolRegistry) -> ToolRegistry:
    """A registry holding one simple sync tool, used by many tests below."""

    @registry.register()
    def add_numbers(a: int, b: int = 10) -> int:
        """Add two integers."""
        return a + b

    return registry


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
def test_register_uses_function_name_and_docstring(registry: ToolRegistry) -> None:
    @registry.register()
    def greet(name: str) -> str:
        """Say hello
        to someone.

        This second paragraph is for humans and is not sent to the AI.
        """
        return f"Hello, {name}!"

    tool = registry.get("greet")

    assert tool is not None
    assert tool.name == "greet"
    assert tool.definition.description == "Say hello to someone."
    assert tool.timeout == DEFAULT_TIMEOUT
    assert tool.is_async is False


def test_decorator_returns_the_original_function(registry: ToolRegistry) -> None:
    @registry.register()
    def double(x: int) -> int:
        """Double a number."""
        return x * 2

    # Still a normal function you can call directly.
    assert double(4) == 8


def test_explicit_name_and_description_win(registry: ToolRegistry) -> None:
    @registry.register(name="custom-name", description="Custom description.")
    def whatever() -> None:
        """Docstring that should be ignored."""

    tool = registry.get("custom-name")

    assert tool is not None
    assert tool.definition.description == "Custom description."
    assert "whatever" not in registry


def test_add_registers_without_decorator(registry: ToolRegistry) -> None:
    def third_party(x: int) -> int:
        return x

    tool = registry.add(third_party, description="Wraps code we do not own.")

    assert registry.get("third_party") is tool


def test_async_tools_are_detected(registry: ToolRegistry) -> None:
    @registry.register()
    async def fetch() -> str:
        """Pretend to fetch something."""
        return "data"

    tool = registry.get("fetch")

    assert tool is not None
    assert tool.is_async is True


def test_duplicate_names_are_rejected(calculator: ToolRegistry) -> None:
    def other() -> None:
        """Another tool."""

    with pytest.raises(ValueError, match="already registered"):
        calculator.add(other, name="add_numbers")


def test_invalid_name_is_rejected(registry: ToolRegistry) -> None:
    def tool() -> None:
        """A tool."""

    with pytest.raises(ValueError, match="name"):
        registry.add(tool, name="has spaces")


def test_missing_description_is_rejected(registry: ToolRegistry) -> None:
    def undocumented() -> None:
        pass

    with pytest.raises(ValueError, match="needs a description"):
        registry.add(undocumented)


@pytest.mark.parametrize("timeout", [0, -1])
def test_non_positive_timeout_is_rejected(
    registry: ToolRegistry, timeout: float
) -> None:
    def tool() -> None:
        """A tool."""

    with pytest.raises(ValueError, match="timeout must be positive"):
        registry.add(tool, timeout=timeout)


def test_missing_type_hint_is_rejected(registry: ToolRegistry) -> None:
    def untyped(x) -> None:  # type: ignore[no-untyped-def]
        """A tool."""

    with pytest.raises(TypeError, match="needs a type hint"):
        registry.add(untyped)


def test_var_args_are_rejected(registry: ToolRegistry) -> None:
    def star_args(*values: int) -> None:
        """A tool."""

    def star_kwargs(**options: int) -> None:
        """A tool."""

    with pytest.raises(TypeError, match=r"\*args/\*\*kwargs"):
        registry.add(star_args)
    with pytest.raises(TypeError, match=r"\*args/\*\*kwargs"):
        registry.add(star_kwargs)


def test_positional_only_parameters_are_rejected(registry: ToolRegistry) -> None:
    # The "/" makes every parameter before it positional-only.
    def positional(x: int, /) -> None:
        """A tool."""

    with pytest.raises(TypeError, match="positional-only"):
        registry.add(positional)


# --------------------------------------------------------------------------- #
# Schema generation
# --------------------------------------------------------------------------- #
def test_schema_is_derived_from_type_hints(registry: ToolRegistry) -> None:
    @registry.register()
    def everything(
        count: int,
        ratio: float,
        label: str,
        enabled: bool,
        tags: list[str],
        mode: Literal["fast", "slow"] = "fast",
        limit: Annotated[int, Field(ge=1, le=10, description="Max items.")] = 5,
        note: str | None = None,
    ) -> None:
        """Use every supported kind of parameter."""

    # Compare against the *exact* JSON the AI will receive.
    assert registry.get_openai_tools() == [
        {
            "type": "function",
            "function": {
                "name": "everything",
                "description": "Use every supported kind of parameter.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer"},
                        "ratio": {"type": "number"},
                        "label": {"type": "string"},
                        "enabled": {"type": "boolean"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "mode": {
                            "type": "string",
                            "enum": ["fast", "slow"],
                            "default": "fast",
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                            "description": "Max items.",
                            "default": 5,
                        },
                        "note": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "default": None,
                        },
                    },
                    "required": ["count", "ratio", "label", "enabled", "tags"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def test_tool_without_parameters_has_empty_schema(registry: ToolRegistry) -> None:
    @registry.register()
    def ping() -> str:
        """Check that tools work."""
        return "pong"

    tool = registry.get("ping")

    assert tool is not None
    assert tool.definition.parameters.model_dump(mode="json") == {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }


def test_get_mcp_tools_uses_input_schema(calculator: ToolRegistry) -> None:
    assert calculator.get_mcp_tools() == [
        {
            "name": "add_numbers",
            "description": "Add two integers.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer", "default": 10},
                },
                "required": ["a"],
                "additionalProperties": False,
            },
        }
    ]


def test_tools_are_listed_in_name_order(registry: ToolRegistry) -> None:
    for name in ("zeta", "alpha", "mid"):
        registry.add(lambda: None, name=name, description="A tool.")

    assert registry.names() == ("alpha", "mid", "zeta")
    assert [d.name for d in registry.definitions()] == ["alpha", "mid", "zeta"]
    assert len(registry) == 3


def test_strip_titles_keeps_properties_and_data_named_title() -> None:
    schema = {
        "title": "drop me",
        "properties": {
            # A *property* called "title" must survive...
            "title": {"title": "drop me too", "type": "string"},
            "meta": {"type": "object", "default": {"title": "keep: data"}},
        },
        "$defs": {"Kind": {"title": "drop", "enum": ["a"]}, "Raw": True},
        "anyOf": [{"title": "drop", "type": "string"}, True],
    }

    assert _strip_titles(schema) == {
        "properties": {
            "title": {"type": "string"},
            "meta": {"type": "object", "default": {"title": "keep: data"}},
        },
        "$defs": {"Kind": {"enum": ["a"]}, "Raw": True},
        "anyOf": [{"type": "string"}, True],
    }


# --------------------------------------------------------------------------- #
# Explicit argument models
# --------------------------------------------------------------------------- #
class SearchArgs(BaseModel):
    query: str = Field(description="Text to search for.")
    limit: int = Field(default=2, ge=1)


def test_args_model_drives_schema_and_execution(registry: ToolRegistry) -> None:
    @registry.register(args_model=SearchArgs)
    def search(args: SearchArgs) -> list[str]:
        """Search for something."""
        return [args.query] * args.limit

    tool = registry.get("search")
    assert tool is not None
    assert tool.definition.parameters.properties == {
        "query": {"type": "string", "description": "Text to search for."},
        "limit": {"type": "integer", "minimum": 1, "default": 2},
    }

    result = registry.execute_sync("search", {"query": "cpu"})

    assert result == ToolExecutionResult.ok(["cpu", "cpu"])


def test_args_model_requires_exactly_one_parameter(registry: ToolRegistry) -> None:
    def two_params(args: SearchArgs, extra: int) -> None:
        """A tool."""

    with pytest.raises(TypeError, match="exactly one parameter"):
        registry.add(two_params, args_model=SearchArgs)


# --------------------------------------------------------------------------- #
# Execution: success
# --------------------------------------------------------------------------- #
async def test_execute_sync_tool(calculator: ToolRegistry) -> None:
    result = await calculator.execute("add_numbers", {"a": 2, "b": 3})

    assert result == ToolExecutionResult.ok(5)


async def test_execute_async_tool(registry: ToolRegistry) -> None:
    @registry.register()
    async def slow_echo(text: str) -> str:
        """Echo text after a short pause."""
        await asyncio.sleep(0.01)
        return text

    result = await registry.execute("slow_echo", {"text": "hi"})

    assert result == ToolExecutionResult.ok("hi")


async def test_defaults_and_type_coercion(calculator: ToolRegistry) -> None:
    # "5" is converted to the int 5 (Pydantic's default "lax" mode), and the
    # missing "b" falls back to its default of 10.
    result = await calculator.execute("add_numbers", {"a": "5"})

    assert result.output == 15


async def test_execute_call_accepts_a_tool_call(calculator: ToolRegistry) -> None:
    call = ToolCall.model_validate(
        {"id": "call_1", "name": "add_numbers", "arguments": '{"a": 1, "b": 1}'}
    )

    result = await calculator.execute_call(call)

    assert result.output == 2


async def test_sync_tools_run_in_a_worker_thread(registry: ToolRegistry) -> None:
    @registry.register()
    def which_thread() -> int:
        """Return the id of the thread running this tool."""
        return threading.get_ident()

    result = await registry.execute("which_thread")

    assert result.output != threading.get_ident()


async def test_output_is_converted_to_json(registry: ToolRegistry) -> None:
    @registry.register()
    def structured() -> tuple[SearchArgs, int]:
        """Return non-JSON Python types that have a JSON form."""
        return SearchArgs(query="x"), 1

    result = await registry.execute("structured")

    assert result.output == [{"query": "x", "limit": 2}, 1]


def test_execute_sync_outside_event_loop(calculator: ToolRegistry) -> None:
    # Note: a plain ``def`` test, so no event loop is running here.
    assert calculator.execute_sync("add_numbers", {"a": 1}).output == 11


async def test_execute_sync_inside_event_loop_raises(calculator: ToolRegistry) -> None:
    with pytest.raises(RuntimeError, match="running event loop"):
        calculator.execute_sync("add_numbers", {"a": 1})


# --------------------------------------------------------------------------- #
# Execution: error containment
# Every case returns an error *result*; nothing is raised.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("arguments", "expected_error"),
    [
        ({}, "Invalid arguments for 'add_numbers': a: Field required"),
        (
            {"a": "not a number"},
            "Invalid arguments for 'add_numbers': a: Input should be a valid "
            "integer, unable to parse string as an integer",
        ),
        (
            {"a": 1, "c": 3},
            "Invalid arguments for 'add_numbers': c: Extra inputs are not permitted",
        ),
    ],
    ids=["missing", "wrong-type", "unexpected"],
)
async def test_invalid_arguments_become_errors(
    calculator: ToolRegistry, arguments: dict[str, Any], expected_error: str
) -> None:
    result = await calculator.execute("add_numbers", arguments)

    assert result == ToolExecutionResult.fail(expected_error)


async def test_unknown_tool(calculator: ToolRegistry) -> None:
    result = await calculator.execute("nope")

    assert result.error == "Unknown tool 'nope'. Available tools: add_numbers"


async def test_unknown_tool_in_empty_registry(registry: ToolRegistry) -> None:
    result = await registry.execute("nope")

    assert result.error == "Unknown tool 'nope'. Available tools: none"


async def test_exceptions_become_errors(registry: ToolRegistry) -> None:
    @registry.register()
    def broken() -> None:
        """Always fails."""
        raise RuntimeError("disk on fire")

    @registry.register()
    def silent_failure() -> None:
        """Fails without a message."""
        raise ValueError

    @registry.register()
    async def broken_async() -> None:
        """Always fails, asynchronously."""
        raise LookupError("not found")

    assert (await registry.execute("broken")).error == "RuntimeError: disk on fire"
    assert (await registry.execute("silent_failure")).error == "ValueError"
    assert (await registry.execute("broken_async")).error == "LookupError: not found"


async def test_sync_tool_timeout(registry: ToolRegistry) -> None:
    @registry.register(timeout=0.05)
    def sleepy() -> None:
        """Sleep longer than allowed."""
        time.sleep(0.3)

    result = await registry.execute("sleepy")

    assert result.error == "Tool 'sleepy' timed out after 0.05 seconds"


async def test_async_tool_timeout(registry: ToolRegistry) -> None:
    @registry.register(timeout=0.05)
    async def sleepy() -> None:
        """Sleep longer than allowed."""
        await asyncio.sleep(5)

    result = await registry.execute("sleepy")

    assert result.error == "Tool 'sleepy' timed out after 0.05 seconds"


async def test_tool_raising_timeout_error_is_not_our_timeout(
    registry: ToolRegistry,
) -> None:
    @registry.register()
    def upstream() -> None:
        """Simulate a network call that timed out inside the tool."""
        raise TimeoutError("upstream server too slow")

    result = await registry.execute("upstream")

    assert result.error == "TimeoutError: upstream server too slow"


async def test_non_json_output_becomes_error(registry: ToolRegistry) -> None:
    @registry.register()
    def weird() -> object:
        """Return something JSON cannot represent."""
        return object()

    result = await registry.execute("weird")

    assert result.error == (
        "Tool 'weird' returned a value that cannot be converted to JSON (object)"
    )


async def test_base_exceptions_are_not_swallowed(registry: ToolRegistry) -> None:
    # KeyboardInterrupt (Ctrl+C) derives from BaseException, not Exception. We
    # use our own BaseException subclass so the test cannot interrupt pytest.
    class StopEverything(BaseException):
        pass

    @registry.register()
    async def interrupt() -> None:
        """Simulate the user pressing Ctrl+C."""
        raise StopEverything

    with pytest.raises(StopEverything):
        await registry.execute("interrupt")


# --------------------------------------------------------------------------- #
# Thread safety
# A Barrier makes all threads wait until everyone is ready and then releases
# them together, which makes collisions as likely as possible.
# --------------------------------------------------------------------------- #
def test_concurrent_registration_of_different_tools(registry: ToolRegistry) -> None:
    thread_count = 50
    barrier = threading.Barrier(thread_count)

    def worker(index: int) -> None:
        barrier.wait()
        registry.add(lambda: index, name=f"tool_{index}", description="A tool.")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(registry) == thread_count


def test_concurrent_registration_of_same_name(registry: ToolRegistry) -> None:
    thread_count = 20
    barrier = threading.Barrier(thread_count)
    outcomes: list[str] = []  # list.append is safe to call from many threads

    def worker() -> None:
        barrier.wait()
        try:
            registry.add(lambda: None, name="shared", description="A tool.")
        except ValueError:
            outcomes.append("rejected")
        else:
            outcomes.append("registered")

    threads = [threading.Thread(target=worker) for _ in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Exactly one thread wins; all the others are told the name is taken.
    assert outcomes.count("registered") == 1
    assert outcomes.count("rejected") == thread_count - 1
