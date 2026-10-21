"""Tool registry: register Python functions as AI tools and run them safely.

Learning notes
--------------
The AI cannot touch your computer. Instead we offer it *tools*: ordinary Python
functions it may ask us to run. This module is the "phone book" of tools:

1. **Register**: put ``@registry.register()`` on a function. The registry reads
   the function's parameters and type hints and works out which arguments the
   tool takes, so you never write a JSON Schema by hand.
2. **Describe**: ``get_openai_tools()`` turns every registered tool into the
   format the AI API expects, so the AI knows what it can ask for.
3. **Execute**: when the AI asks for a tool, ``execute()`` looks it up, checks
   the arguments, runs it, and *always* returns a ``ToolExecutionResult``. Bad
   arguments, crashes and timeouts become error results instead of crashing
   the CLI, because the AI will sometimes make mistakes.

Key idea: *one source of truth*. The function signature produces both the
schema we show the AI and the validator we check its arguments with, so the two
can never disagree.

What this does NOT protect against: everything runs in the same process, so a
tool that calls ``os._exit()``, eats all memory, or crashes inside C code still
takes the CLI down. Catching exceptions is *containment*, not *isolation*; real
isolation would mean running tools in a separate process.
"""

import asyncio
import functools
import inspect
import logging
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, ParamSpec, TypeVar, get_type_hints

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError, create_model
from pydantic_core import PydanticSerializationError, to_jsonable_python

from lightspeed_cli.schemas import (
    ToolCall,
    ToolDefinition,
    ToolExecutionResult,
    ToolParameterSchema,
)

# ``logging`` is Python's standard way to record diagnostic messages. Naming the
# logger after the module (``lightspeed_cli.tools.base``) lets users turn
# messages from this module up or down independently.
logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30.0
"""Seconds a tool may run before ``execute()`` gives up on it."""

# ParamSpec and TypeVar let the decorator say "I return a function with exactly
# the same parameters (P) and return type (R) as the one you gave me". Without
# them, mypy would forget the decorated function's signature.
P = ParamSpec("P")
R = TypeVar("R")

# JSON Schema keywords whose values are *data*, not nested schemas. When we tidy
# up a schema we must not touch these (a default value may legitimately contain
# a key called "title").
_SCHEMA_DATA_KEYWORDS = frozenset({"default", "enum", "const", "examples"})


# --------------------------------------------------------------------------- #
# RegisteredTool: everything the registry knows about one tool
# --------------------------------------------------------------------------- #
# ``@dataclass`` writes ``__init__``, ``__repr__`` and ``__eq__`` for us from the
# field list. frozen=True makes instances read-only (like our Pydantic models);
# slots=True makes them a little smaller and faster. We use a dataclass rather
# than Pydantic here because this is internal data that needs no validation.
@dataclass(frozen=True, slots=True)
class RegisteredTool:
    """A registered tool: its public definition plus what is needed to run it."""

    definition: ToolDefinition
    """What the AI sees: name, description, parameter schema."""

    func: Callable[..., Any]
    """The original Python function."""

    args_model: type[BaseModel]
    """Pydantic model used to validate the AI's arguments before calling."""

    is_async: bool
    """True for ``async def`` tools, which are awaited instead of threaded."""

    timeout: float
    """Seconds before the call is abandoned."""

    takes_model: bool
    """True if ``func`` receives one model object instead of keyword arguments."""

    @property
    def name(self) -> str:
        # ``@property`` lets callers write ``tool.name`` (no parentheses) even
        # though the value is computed.
        return self.definition.name


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #
class ToolRegistry:
    """A thread-safe collection of tools that the AI may call.

    Example::

        registry = ToolRegistry()

        @registry.register()
        def add(a: int, b: int) -> int:
            \"\"\"Add two integers.\"\"\"
            return a + b

        result = registry.execute_sync("add", {"a": 2, "b": 3})
        assert result.output == 5
    """

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        # A lock lets only one thread at a time into a ``with self._lock:``
        # block. Without it, two threads registering the same name at the same
        # moment could both pass the "already registered?" check. A plain Lock
        # is enough because no method takes the lock twice.
        self._lock = threading.Lock()

    # ----------------------------------------------------------------------- #
    # Registration
    # ----------------------------------------------------------------------- #
    def register(
        self,
        name: str | None = None,
        description: str | None = None,
        *,
        args_model: type[BaseModel] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Decorator that registers a function as a tool.

        Args:
            name: Tool name. Defaults to the function's name.
            description: Text for the AI. Defaults to the docstring's first
                paragraph.
            args_model: Optional Pydantic model describing the arguments. The
                function must then take exactly one parameter, which receives
                the validated model. Without it, arguments are derived from the
                function's own parameters.
            timeout: Seconds the tool may run.

        Note the parentheses: write ``@registry.register()``, not
        ``@registry.register``. ``register(...)`` *returns* the decorator.
        """

        # A decorator with options is a function that returns a decorator.
        # ``decorator`` below is what Python actually applies to the function.
        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            self.add(
                func,
                name=name,
                description=description,
                args_model=args_model,
                timeout=timeout,
            )
            # Return the original function unchanged, so it can still be called
            # directly (handy in tests and in other Python code).
            return func

        return decorator

    def add(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        args_model: type[BaseModel] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> RegisteredTool:
        """Register ``func`` without decorator syntax and return its record.

        Raises:
            ValueError: invalid name, missing description, non-positive timeout,
                or a tool with this name already exists.
            TypeError: the function signature cannot be exposed as a tool.
        """
        # Do all the slow and failure-prone work *before* taking the lock, so
        # the lock is held for as short a time as possible.
        tool = _build_tool(func, name, description, args_model, timeout)
        with self._lock:
            if tool.name in self._tools:
                raise ValueError(f"A tool named {tool.name!r} is already registered")
            self._tools[tool.name] = tool
        return tool

    # ----------------------------------------------------------------------- #
    # Looking things up
    # ----------------------------------------------------------------------- #
    def get(self, name: str) -> RegisteredTool | None:
        """Return the tool called ``name``, or ``None``."""
        with self._lock:
            return self._tools.get(name)

    def names(self) -> tuple[str, ...]:
        """Return all tool names, sorted."""
        return tuple(tool.name for tool in self._snapshot())

    def definitions(self) -> tuple[ToolDefinition, ...]:
        """Return all tool definitions, sorted by name."""
        return tuple(tool.definition for tool in self._snapshot())

    def get_openai_tools(self) -> list[dict[str, JsonValue]]:
        """Return all tools in the format of OpenAI's ``tools=[...]`` parameter."""
        return [definition.to_openai_tool() for definition in self.definitions()]

    def get_mcp_tools(self) -> list[dict[str, JsonValue]]:
        """Return all tools in the format of an MCP ``tools/list`` response."""
        return [definition.to_mcp_tool() for definition in self.definitions()]

    # "Dunder" (double underscore) methods hook into Python syntax:
    # ``"x" in registry`` calls __contains__ and ``len(registry)`` calls __len__.
    def __contains__(self, name: object) -> bool:
        with self._lock:
            return name in self._tools

    def __len__(self) -> int:
        with self._lock:
            return len(self._tools)

    def _snapshot(self) -> list[RegisteredTool]:
        # Copy the tools while holding the lock, then work on the copy without
        # it. Sorting gives stable output, so the AI sees tools in the same order
        # every time and tests can compare exact lists.
        with self._lock:
            tools = list(self._tools.values())
        return sorted(tools, key=lambda tool: tool.name)

    # ----------------------------------------------------------------------- #
    # Execution
    # ----------------------------------------------------------------------- #
    async def execute(
        self, name: str, arguments: Mapping[str, JsonValue] | None = None
    ) -> ToolExecutionResult:
        """Validate ``arguments``, run the tool, and report what happened.

        This never raises because of a tool problem: every failure comes back
        as ``ToolExecutionResult(status="error")`` with a short message meant
        for the AI. Only ``BaseException`` subclasses that are not
        ``Exception`` (Ctrl+C, task cancellation) pass through.
        """
        # ``async def`` makes this a *coroutine function*: calling it returns a
        # coroutine that runs when someone ``await``s it. While one coroutine
        # waits (for a timer, a thread, the network), the event loop can run
        # others, so a slow tool does not freeze the whole program.

        # Step 1: look the tool up.
        tool = self.get(name)
        if tool is None:
            available = ", ".join(self.names()) or "none"
            return ToolExecutionResult.fail(
                f"Unknown tool {name!r}. Available tools: {available}"
            )

        # Step 2: validate the arguments *before* running anything.
        try:
            validated = tool.args_model.model_validate(dict(arguments or {}))
        except ValidationError as exc:
            return ToolExecutionResult.fail(
                f"Invalid arguments for {name!r}: {_describe_validation_error(exc)}"
            )

        # Step 3: run it, with a time limit. We keep a reference to the timeout
        # object so we can later tell *our* timeout apart from a TimeoutError
        # that the tool itself raised (e.g. a network call inside the tool).
        deadline = asyncio.timeout(tool.timeout)
        try:
            async with deadline:
                output = await _invoke(tool, validated)
        except TimeoutError as exc:
            if deadline.expired():
                return ToolExecutionResult.fail(
                    f"Tool {name!r} timed out after {tool.timeout:g} seconds"
                )
            return _failure_from_exception(name, exc)
        # Catch ``Exception``, never bare ``except:`` or ``BaseException``:
        # those would also swallow KeyboardInterrupt, and Ctrl+C would stop
        # working while a tool runs.
        except Exception as exc:
            return _failure_from_exception(name, exc)

        # Step 4: make sure the output can be sent to the AI as JSON.
        return _success_from_output(name, output)

    async def execute_call(self, call: ToolCall) -> ToolExecutionResult:
        """Execute a ``ToolCall`` exactly as received from the AI."""
        return await self.execute(call.name, call.arguments)

    def execute_sync(
        self, name: str, arguments: Mapping[str, JsonValue] | None = None
    ) -> ToolExecutionResult:
        """Blocking version of :meth:`execute` for non-async code.

        Raises:
            RuntimeError: when called from inside a running event loop. There
                you must ``await registry.execute(...)`` instead, because
                ``asyncio.run`` cannot start a second loop inside the first.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No loop is running (the normal case in a plain script or a Typer
            # command), so start one just for this call.
            return asyncio.run(self.execute(name, arguments))
        raise RuntimeError(
            "execute_sync() cannot be called from a running event loop; "
            "use 'await registry.execute(...)' instead"
        )


# --------------------------------------------------------------------------- #
# Building a RegisteredTool from a function
# --------------------------------------------------------------------------- #
def _build_tool(
    func: Callable[..., Any],
    name: str | None,
    description: str | None,
    args_model: type[BaseModel] | None,
    timeout: float,
) -> RegisteredTool:
    if timeout <= 0:
        raise ValueError(f"timeout must be positive, got {timeout}")

    tool_name = name or func.__name__
    tool_description = description or _first_paragraph(inspect.getdoc(func))
    if not tool_description:
        raise ValueError(
            f"Tool {tool_name!r} needs a description: add a docstring or pass "
            "description=..."
        )

    if args_model is None:
        model = _args_model_from_signature(func, tool_name)
        takes_model = False
    else:
        _check_single_parameter(func, tool_name)
        model = args_model
        takes_model = True

    # Building the ToolDefinition also validates the name against
    # TOOL_NAME_PATTERN. Pydantic's ValidationError is a subclass of ValueError,
    # so callers can catch a bad name with ``except ValueError``.
    definition = ToolDefinition(
        name=tool_name,
        description=tool_description,
        parameters=_parameter_schema(model),
    )
    return RegisteredTool(
        definition=definition,
        func=func,
        args_model=model,
        is_async=inspect.iscoroutinefunction(func),
        timeout=timeout,
        takes_model=takes_model,
    )


def _first_paragraph(docstring: str | None) -> str:
    """Return the docstring's first paragraph, joined onto one line."""
    if not docstring:
        return ""
    paragraph = docstring.split("\n\n", maxsplit=1)[0]
    # split() with no argument splits on any whitespace (including newlines);
    # joining with " " collapses a wrapped paragraph onto one line.
    return " ".join(paragraph.split())


def _args_model_from_signature(
    func: Callable[..., Any], tool_name: str
) -> type[BaseModel]:
    """Build a Pydantic model whose fields mirror ``func``'s parameters.

    For ``def get_cpu(interval: float = 1.0)`` this is equivalent to writing::

        class get_cpu_arguments(BaseModel):
            model_config = ConfigDict(extra="forbid")
            interval: float = 1.0
    """
    # ``inspect.signature`` describes the parameters (names, kinds, defaults).
    # ``get_type_hints`` resolves their type hints; include_extras=True keeps
    # ``Annotated[...]`` metadata such as Field(description=..., ge=...).
    signature = inspect.signature(func)
    hints = get_type_hints(func, include_extras=True)

    fields: dict[str, Any] = {}
    for param in signature.parameters.values():
        # The AI sends arguments *by name* as a JSON object, so every parameter
        # must have a name we can show it and a type we can validate.
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise TypeError(
                f"Tool {tool_name!r}: *args/**kwargs cannot be described to the "
                f"AI; declare each parameter explicitly ({param.name!r})"
            )
        if param.kind is param.POSITIONAL_ONLY:
            raise TypeError(
                f"Tool {tool_name!r}: positional-only parameter {param.name!r} "
                "cannot be passed by name"
            )
        if param.name not in hints:
            raise TypeError(
                f"Tool {tool_name!r}: parameter {param.name!r} needs a type hint"
            )
        # In create_model, ``...`` (Ellipsis) means "required, no default".
        default = ... if param.default is param.empty else param.default
        fields[param.name] = (hints[param.name], default)

    # create_model builds a class at runtime, same as writing it out by hand.
    # extra="forbid": arguments the function does not accept are errors, rather
    # than being silently dropped. It also makes the schema say
    # "additionalProperties": false.
    return create_model(
        f"{tool_name}_arguments",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


def _check_single_parameter(func: Callable[..., Any], tool_name: str) -> None:
    if len(inspect.signature(func).parameters) != 1:
        raise TypeError(
            f"Tool {tool_name!r}: a tool registered with args_model must take "
            "exactly one parameter (the validated model)"
        )


def _parameter_schema(model: type[BaseModel]) -> ToolParameterSchema:
    """Turn a Pydantic model into the JSON Schema we show the AI."""
    # model_json_schema() writes the JSON Schema for us. Pydantic adds a
    # "title" to every schema (e.g. "Interval"); they carry no information for
    # the AI and cost tokens, so we strip them.
    return ToolParameterSchema.model_validate(_strip_titles(model.model_json_schema()))


def _strip_titles(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively remove ``title`` keywords from a JSON Schema.

    Only *keywords* are removed: a property that happens to be called "title"
    (a key inside "properties") and data such as default values are kept.
    """
    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "title" and isinstance(value, str):
            continue  # a title keyword: drop it
        if key in _SCHEMA_DATA_KEYWORDS:
            cleaned[key] = value  # data, not a schema: copy untouched
        elif key in ("properties", "$defs") and isinstance(value, dict):
            # A mapping of {name: schema}: keep every name, clean each schema.
            cleaned[key] = {
                prop: _strip_titles(sub) if isinstance(sub, dict) else sub
                for prop, sub in value.items()
            }
        elif isinstance(value, dict):
            cleaned[key] = _strip_titles(value)  # e.g. "items": {...}
        elif isinstance(value, list):
            # e.g. "anyOf": [{...}, {...}]
            cleaned[key] = [
                _strip_titles(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned


# --------------------------------------------------------------------------- #
# Running a tool and turning the outcome into a ToolExecutionResult
# --------------------------------------------------------------------------- #
async def _invoke(tool: RegisteredTool, validated: BaseModel) -> object:
    if tool.takes_model:
        call = functools.partial(tool.func, validated)
    else:
        # Read the validated values as attributes (not model_dump()), so they
        # keep their real Python types: a ``Path`` parameter receives a Path.
        kwargs = {
            field: getattr(validated, field) for field in type(validated).model_fields
        }
        # functools.partial "pre-fills" arguments: call() == tool.func(**kwargs)
        call = functools.partial(tool.func, **kwargs)

    if tool.is_async:
        return await call()
    # A normal (sync) function would block the event loop while it runs, for
    # example psutil.cpu_percent(interval=1) sleeps for a whole second.
    # asyncio.to_thread runs it in a worker thread and lets us await the result.
    # Caveat: Python cannot kill a thread, so on timeout the thread keeps
    # running in the background until the function returns by itself.
    return await asyncio.to_thread(call)


def _success_from_output(name: str, output: object) -> ToolExecutionResult:
    # to_jsonable_python converts tuples, Pydantic models, dataclasses, dates,
    # paths, ... into plain JSON-compatible data, and raises if it cannot.
    try:
        jsonable = to_jsonable_python(output)
    except PydanticSerializationError:
        return ToolExecutionResult.fail(
            f"Tool {name!r} returned a value that cannot be converted to JSON "
            f"({type(output).__name__})"
        )
    return ToolExecutionResult.ok(jsonable)


def _failure_from_exception(name: str, exc: Exception) -> ToolExecutionResult:
    # The full traceback is useful to a developer, so log it at DEBUG level.
    # The AI only gets a short message: tracebacks waste tokens and can reveal
    # file paths on your machine.
    logger.debug("Tool %r raised an exception", name, exc_info=exc)
    message = str(exc)
    kind = type(exc).__name__
    return ToolExecutionResult.fail(f"{kind}: {message}" if message else kind)


def _describe_validation_error(exc: ValidationError) -> str:
    """Summarise a ValidationError as "field: problem; field: problem"."""
    # exc.errors() is a list of dicts like
    # {"loc": ("interval",), "msg": "Input should be greater than 0", ...}
    problems = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "arguments"
        problems.append(f"{location}: {error['msg']}")
    return "; ".join(problems)


# --------------------------------------------------------------------------- #
# The application's default registry. Built-in tools register themselves on it
# (see tools/system.py). Tests create their own ToolRegistry() instead, so they
# never interfere with each other or with the built-in tools.
# --------------------------------------------------------------------------- #
registry = ToolRegistry()
