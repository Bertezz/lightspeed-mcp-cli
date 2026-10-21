"""Application settings loaded from the environment and an optional ``.env`` file.

Learning notes
--------------
* Programs should not hard-code things like API keys or server addresses. They
  read them from *environment variables* (``export LOG_LEVEL=DEBUG``) or from a
  ``.env`` file, so every user can configure the program without editing code.
* ``pydantic-settings`` does the boring part for us: we declare fields with type
  hints and defaults, and it finds the values, converts text such as ``"12.5"``
  to the right type (``12.5``), and *validates* them (rejects ``-5`` seconds).
* The result is one trustworthy ``Settings`` object. If it was created without
  an error, every value in it is known to be valid.
"""

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, HttpUrl, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ``Literal[...]`` means "exactly one of these values". Giving the type a name
# (a *type alias*) lets us reuse it and makes signatures easier to read.
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class Settings(BaseSettings):
    """Runtime configuration for the ``lightspeed`` CLI.

    Values resolve in this order (highest priority first): keyword arguments,
    environment variables, the ``.env`` file in the working directory, defaults.
    Empty values (e.g. ``OPENAI_BASE_URL=``) are ignored and fall back to defaults.
    """

    # ``model_config`` is a special class attribute: it configures how Pydantic
    # treats this class rather than being a setting itself.
    model_config = SettingsConfigDict(
        env_file=".env",  # also read KEY=value lines from ./.env
        env_file_encoding="utf-8",
        env_ignore_empty=True,  # "KEY=" (empty) counts as "not set"
        case_sensitive=False,  # log_level and LOG_LEVEL are the same variable
        extra="ignore",  # unrelated keys in .env are not errors
        frozen=True,  # settings cannot be changed after loading
        validate_by_name=True,  # allow Settings(default_model=...) despite aliases
    )

    # Each line below is a *field*: ``name: type = default``. The environment
    # variable name is the field name in upper case (OPENAI_API_KEY).

    openai_api_key: SecretStr | None = None
    """API key; optional because local engines (Ollama, vLLM) do not need one.

    ``SecretStr`` hides the value when printed (``**********``), so the key does
    not leak into logs or screenshots by accident. Read the real value with
    ``settings.openai_api_key.get_secret_value()``, which is easy to search for.
    """

    openai_base_url: HttpUrl | None = None
    """OpenAI-compatible endpoint; ``None`` means the SDK default.

    ``HttpUrl`` rejects typos such as ``localhost:11434`` (no ``http://``) at
    start-up, instead of failing with a confusing network error later.
    """

    # ``Field(...)`` attaches extra rules to a field. ``AliasChoices`` accepts
    # either environment variable name; the first one found wins, so the
    # unambiguous LIGHTSPEED_ name beats a generic name another program might use.
    default_model: str = Field(
        default="gpt-4o-mini",
        min_length=1,
        validation_alias=AliasChoices("LIGHTSPEED_DEFAULT_MODEL", "DEFAULT_MODEL"),
    )
    log_level: LogLevel = Field(
        default="INFO",
        validation_alias=AliasChoices("LIGHTSPEED_LOG_LEVEL", "LOG_LEVEL"),
    )
    # gt = "greater than", le = "less than or equal": 0 < timeout <= 600.
    request_timeout: float = Field(
        default=30.0,
        gt=0,
        le=600,
        validation_alias=AliasChoices("LIGHTSPEED_REQUEST_TIMEOUT", "REQUEST_TIMEOUT"),
    )
    # How many request -> tool calls -> request rounds one question may take
    # before the assistant is stopped (protects against endless tool loops).
    max_tool_iterations: int = Field(
        default=5,
        ge=1,
        le=20,
        validation_alias=AliasChoices(
            "LIGHTSPEED_MAX_TOOL_ITERATIONS", "MAX_TOOL_ITERATIONS"
        ),
    )
    # How often the openai SDK itself retries a failed request (rate limits,
    # server errors, dropped connections), waiting a little longer each time.
    llm_max_retries: int = Field(
        default=2,
        ge=0,
        le=10,
        validation_alias=AliasChoices("LIGHTSPEED_LLM_MAX_RETRIES", "LLM_MAX_RETRIES"),
    )

    # ----------------------------------------------------------------------- #
    # Validators: small functions Pydantic calls while building the object.
    # ``mode="before"`` runs them on the raw input, before the type checks and
    # Field rules above. Validators must be classmethods: they run before the
    # object exists, so there is no ``self`` yet.
    # The leading underscore means "internal; not for use outside this class".
    # ----------------------------------------------------------------------- #
    @field_validator("default_model", mode="before")
    @classmethod
    def _strip_model(cls, value: object) -> object:
        # "  llama3  " -> "llama3". Whitespace-only becomes "", which the
        # min_length=1 rule then rejects. Non-strings are left for Pydantic to
        # reject with its normal error message.
        return value.strip() if isinstance(value, str) else value

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_log_level(cls, value: object) -> object:
        # Accept "debug" or " Debug " by normalising to "DEBUG" before the
        # Literal check runs.
        return value.strip().upper() if isinstance(value, str) else value

    def base_url_str(self) -> str | None:
        """Return the base URL as a plain string for the OpenAI client."""
        # The ``openai`` library wants a plain ``str``, not Pydantic's HttpUrl.
        return str(self.openai_base_url) if self.openai_base_url else None


# ``@lru_cache`` remembers a function's return value. The first call builds
# ``Settings()`` (reading the environment and .env); every later call returns
# the *same* object instantly. One shared instance like this is a "singleton".
# Tests call ``get_settings.cache_clear()`` to force a fresh load.
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, loading them on first use."""
    return Settings()
