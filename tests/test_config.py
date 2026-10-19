"""Tests for settings resolution and validation."""

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from lightspeed_cli.config import Settings, get_settings


def write_dotenv(directory: Path, content: str) -> None:
    (directory / ".env").write_text(content, encoding="utf-8")


def test_defaults() -> None:
    settings = Settings()

    assert settings.openai_api_key is None
    assert settings.openai_base_url is None
    assert settings.base_url_str() is None
    assert settings.default_model == "gpt-4o-mini"
    assert settings.log_level == "INFO"
    assert settings.request_timeout == 30.0


def test_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("DEFAULT_MODEL", "llama3")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("REQUEST_TIMEOUT", "12.5")

    settings = Settings()

    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "sk-test"
    assert settings.base_url_str() == "http://localhost:11434/v1"
    assert settings.default_model == "llama3"
    assert settings.log_level == "DEBUG"
    assert settings.request_timeout == 12.5


def test_keyword_arguments_override_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEFAULT_MODEL", "from-env")

    settings = Settings(default_model="from-kwargs", openai_api_key=SecretStr("k"))

    assert settings.default_model == "from-kwargs"


def test_env_names_are_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("default_model", "lowercase-name")

    assert Settings().default_model == "lowercase-name"


def test_prefixed_alias_wins_over_bare_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEFAULT_MODEL", "bare")
    monkeypatch.setenv("LIGHTSPEED_DEFAULT_MODEL", "prefixed")
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    monkeypatch.setenv("LIGHTSPEED_LOG_LEVEL", "WARNING")

    settings = Settings()

    assert settings.default_model == "prefixed"
    assert settings.log_level == "WARNING"


def test_dotenv_is_loaded(isolated_env: Path) -> None:
    write_dotenv(isolated_env, "DEFAULT_MODEL=from-dotenv\nREQUEST_TIMEOUT=5\n")

    settings = Settings()

    assert settings.default_model == "from-dotenv"
    assert settings.request_timeout == 5.0


def test_environment_beats_dotenv(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_dotenv(isolated_env, "DEFAULT_MODEL=from-dotenv\n")
    monkeypatch.setenv("DEFAULT_MODEL", "from-env")

    assert Settings().default_model == "from-env"


def test_empty_values_fall_back_to_defaults(isolated_env: Path) -> None:
    write_dotenv(
        isolated_env,
        "OPENAI_API_KEY=\nOPENAI_BASE_URL=\nDEFAULT_MODEL=\nLOG_LEVEL=\n",
    )

    settings = Settings()

    assert settings.openai_api_key is None
    assert settings.openai_base_url is None
    assert settings.default_model == "gpt-4o-mini"
    assert settings.log_level == "INFO"


def test_unknown_dotenv_keys_are_ignored(isolated_env: Path) -> None:
    write_dotenv(isolated_env, "SOME_OTHER_TOOL_SETTING=1\n")

    assert Settings().default_model == "gpt-4o-mini"


def test_api_key_is_never_rendered(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "sk-super-secret-value"
    monkeypatch.setenv("OPENAI_API_KEY", secret)

    settings = Settings()

    assert secret not in repr(settings)
    assert secret not in str(settings)
    assert secret not in settings.model_dump_json()
    assert secret not in f"{settings.openai_api_key}"
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == secret


@pytest.mark.parametrize(
    "url",
    ["http://localhost:11434/v1", "https://api.example.com/v1", "http://10.0.0.5:8000"],
)
def test_base_url_accepts_http_and_https(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", url)

    base_url = Settings().base_url_str()

    assert base_url is not None
    assert base_url.startswith(url)


@pytest.mark.parametrize("url", ["not-a-url", "ftp://example.com", "localhost:11434"])
def test_base_url_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", url)

    with pytest.raises(ValidationError):
        Settings()


def test_default_model_is_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEFAULT_MODEL", "  llama3  ")

    assert Settings().default_model == "llama3"


def test_default_model_rejects_whitespace_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEFAULT_MODEL", "   ")

    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("debug", "DEBUG"), ("Warning", "WARNING"), (" error ", "ERROR")],
)
def test_log_level_is_normalized(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: str
) -> None:
    monkeypatch.setenv("LOG_LEVEL", raw)

    assert Settings().log_level == expected


def test_log_level_rejects_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "TRACE")

    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize(("raw", "expected"), [("0.5", 0.5), ("600", 600.0)])
def test_request_timeout_accepts_bounds(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: float
) -> None:
    monkeypatch.setenv("REQUEST_TIMEOUT", raw)

    assert Settings().request_timeout == expected


@pytest.mark.parametrize("raw", ["0", "-1", "600.1", "abc"])
def test_request_timeout_rejects_out_of_range(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv("REQUEST_TIMEOUT", raw)

    with pytest.raises(ValidationError):
        Settings()


def test_settings_are_frozen() -> None:
    settings = Settings()

    with pytest.raises(ValidationError):
        settings.default_model = "other"  # type: ignore[misc]


def test_get_settings_is_cached() -> None:
    first = get_settings()

    assert get_settings() is first

    get_settings.cache_clear()

    assert get_settings() is not first
