"""Application settings loaded from the environment and an optional ``.env`` file."""

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, HttpUrl, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class Settings(BaseSettings):
    """Runtime configuration for the ``lightspeed`` CLI.

    Values resolve in this order (highest priority first): keyword arguments,
    environment variables, the ``.env`` file in the working directory, defaults.
    Empty values (e.g. ``OPENAI_BASE_URL=``) are ignored and fall back to defaults.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        case_sensitive=False,
        extra="ignore",
        frozen=True,
        validate_by_name=True,
    )

    openai_api_key: SecretStr | None = None
    """API key; optional because local engines (Ollama, vLLM) do not need one."""

    openai_base_url: HttpUrl | None = None
    """OpenAI-compatible endpoint; ``None`` means the SDK default."""

    default_model: str = Field(
        default="gpt-4o-mini",
        min_length=1,
        validation_alias=AliasChoices("LIGHTSPEED_DEFAULT_MODEL", "DEFAULT_MODEL"),
    )
    log_level: LogLevel = Field(
        default="INFO",
        validation_alias=AliasChoices("LIGHTSPEED_LOG_LEVEL", "LOG_LEVEL"),
    )
    request_timeout: float = Field(
        default=30.0,
        gt=0,
        le=600,
        validation_alias=AliasChoices("LIGHTSPEED_REQUEST_TIMEOUT", "REQUEST_TIMEOUT"),
    )

    @field_validator("default_model", mode="before")
    @classmethod
    def _strip_model(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_log_level(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    def base_url_str(self) -> str | None:
        """Return the base URL as a plain string for the OpenAI client."""
        return str(self.openai_base_url) if self.openai_base_url else None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, loading them on first use."""
    return Settings()
