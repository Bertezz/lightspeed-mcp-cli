"""Shared test fixtures.

Learning notes
--------------
pytest automatically loads ``conftest.py`` before running the tests in this
folder. Fixtures defined here are available to every test file.

A *fixture* is a function that prepares something a test needs. A test asks for
a fixture simply by having a parameter with the fixture's name; pytest calls the
fixture and passes the result in. ``monkeypatch`` and ``tmp_path`` used below
are fixtures that pytest itself provides.
"""

import logging
import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import pytest
from pytest_mock import MockerFixture

from lightspeed_cli.client import ToolEvent
from lightspeed_cli.config import get_settings
from lightspeed_cli.logging_setup import APP_LOGGER, QUIET_LOGGERS
from lightspeed_cli.schemas import ChatMessage, ChatResult, TokenUsage

_SETTINGS_ENV_VARS = frozenset(
    {
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "DEFAULT_MODEL",
        "LOG_LEVEL",
        "REQUEST_TIMEOUT",
        "LIGHTSPEED_DEFAULT_MODEL",
        "LIGHTSPEED_LOG_LEVEL",
        "LIGHTSPEED_REQUEST_TIMEOUT",
        "MAX_TOOL_ITERATIONS",
        "LIGHTSPEED_MAX_TOOL_ITERATIONS",
        "LLM_MAX_RETRIES",
        "LIGHTSPEED_LLM_MAX_RETRIES",
    }
)


# autouse=True -> this fixture runs around *every* test, even tests that do not
# ask for it. Without it, a test could pass on your laptop but fail on a
# colleague's machine, just because their environment or .env file differs.
@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """Hide the developer's environment and ``.env`` from every test.

    Yields an empty working directory in which tests may write a ``.env`` file.
    """
    # monkeypatch changes things temporarily and undoes them after the test.
    for name in list(os.environ):
        if name.upper() in _SETTINGS_ENV_VARS:
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)  # an empty temp folder: no real .env here
    get_settings.cache_clear()
    # Everything before ``yield`` is setup; the test runs at the ``yield``;
    # everything after it is cleanup (it runs even if the test fails).
    yield tmp_path
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def reset_logging() -> Iterator[None]:
    """Undo configure_logging() after each test.

    Loggers are global objects: a handler added in one test would otherwise
    still be attached (and printing) in every later test.
    """
    yield
    for name in (APP_LOGGER, *QUIET_LOGGERS):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.setLevel(logging.NOTSET)
        logger.propagate = True


# --------------------------------------------------------------------------- #
# A fake AssistantClient for CLI tests
# --------------------------------------------------------------------------- #
def make_result(
    prompt: str, history: Sequence[ChatMessage] = (), reply: str | None = None
) -> ChatResult:
    """A ChatResult like the real client returns for a simple answer."""
    answer = reply if reply is not None else f"Echo: {prompt}"
    return ChatResult(
        reply=answer,
        messages=(*history, ChatMessage.user(prompt), ChatMessage.assistant(answer)),
        usage=TokenUsage(prompt_tokens=8, completion_tokens=4, total_tokens=12),
        iterations=1,
    )


@dataclass
class FakeAssistantControl:
    """Lets a test script the fake client and inspect how it was used.

    ``outcomes`` holds one entry per chat() call: a ChatResult to return or an
    exception to raise. When it runs out, chat() echoes the prompt.
    """

    outcomes: list[ChatResult | BaseException] = field(default_factory=list)
    events: list[ToolEvent] = field(default_factory=list)  # sent on every chat()
    created: list[dict[str, Any]] = field(default_factory=list)  # constructor args
    prompts: list[str] = field(default_factory=list)
    histories: list[tuple[ChatMessage, ...]] = field(default_factory=list)
    closed: int = 0


@pytest.fixture
def fake_assistant(mocker: MockerFixture) -> FakeAssistantControl:
    """Replace AssistantClient in main.py with a scriptable fake.

    ``mocker.patch`` swaps the *name* ``AssistantClient`` inside
    ``lightspeed_cli.main``, so the commands build our fake instead of a real
    client, and no test ever talks to a model server.
    """
    control = FakeAssistantControl()

    class FakeAssistant:
        def __init__(
            self,
            settings: object = None,
            registry: object = None,
            *,
            model: str | None = None,
            temperature: float | None = None,
            on_event: Callable[[ToolEvent], None] | None = None,
        ) -> None:
            control.created.append({"model": model, "temperature": temperature})
            self.model = model or "fake-model"
            self.base_url = "http://fake.test/v1/"
            self._on_event = on_event

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            await self.aclose()

        async def aclose(self) -> None:
            control.closed += 1

        async def chat(
            self, prompt: str, history: Sequence[ChatMessage] = ()
        ) -> ChatResult:
            control.prompts.append(prompt)
            control.histories.append(tuple(history))
            if self._on_event is not None:
                for event in control.events:
                    self._on_event(event)
            if not control.outcomes:
                return make_result(prompt, history)
            outcome = control.outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    mocker.patch("lightspeed_cli.main.AssistantClient", FakeAssistant)
    return control
