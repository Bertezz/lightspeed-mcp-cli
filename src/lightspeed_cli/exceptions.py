"""Exceptions raised by lightspeed-mcp-cli.

Learning notes
--------------
Exceptions form a *hierarchy*: ``except LLMError`` also catches every subclass
of ``LLMError``. That lets callers choose how specific to be:

* the CLI catches ``LightspeedError`` (everything) and prints a friendly message;
* other code could catch just ``LLMRateLimitError`` and wait before retrying.

Why our own exceptions instead of the ``openai`` package's? The rest of the
program then never depends on the SDK's exception classes, which change
between SDK versions. ``client.py`` is the single place that translates them.

Every message is written for the *user* (what went wrong and what to try),
and never contains the API key.

Class names start with ``LLM`` on purpose: ``openai`` already has classes such
as ``APIConnectionError``, and two classes with the same name in one file is a
classic source of confusing bugs.
"""

from lightspeed_cli.schemas import ChatMessage


class LightspeedError(Exception):
    """Base class for every error this application raises on purpose."""


class ConfigurationError(LightspeedError):
    """The settings are incomplete or inconsistent (e.g. a missing API key)."""


# --------------------------------------------------------------------------- #
# Problems talking to the model server
# --------------------------------------------------------------------------- #
class LLMError(LightspeedError):
    """Talking to the model server failed."""

    def __init__(self, message: str, *, provider_message: str | None = None) -> None:
        super().__init__(message)
        # The server's own explanation (e.g. "model 'x' not found"), if any.
        # Keyword-only (after the ``*``), so calls read clearly at the call site.
        self.provider_message = provider_message


class LLMConnectionError(LLMError):
    """The model server could not be reached (wrong URL, server not running)."""


class LLMTimeoutError(LLMError):
    """The model server did not answer within ``request_timeout`` seconds."""


class LLMRateLimitError(LLMError):
    """The provider kept refusing requests because too many were sent."""


class LLMAuthenticationError(LLMError):
    """The API key is missing, wrong, or not allowed to use this model."""


class LLMResponseError(LLMError):
    """The server rejected the request or sent back something unusable."""


# --------------------------------------------------------------------------- #
# Problems in the conversation itself
# --------------------------------------------------------------------------- #
class MaxIterationsExceededError(LightspeedError):
    """The model kept requesting tools without ever giving a final answer."""

    def __init__(self, iterations: int, messages: tuple[ChatMessage, ...]) -> None:
        super().__init__(
            f"The assistant was still calling tools after {iterations} rounds, so it "
            "was stopped to avoid an endless loop. Try a more specific question, or "
            "raise LIGHTSPEED_MAX_TOOL_ITERATIONS."
        )
        self.iterations = iterations
        # The conversation so far, so a caller can still show which tools ran.
        self.messages = messages
