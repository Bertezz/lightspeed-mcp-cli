"""Tests for ``lightspeed version`` and ``lightspeed --version``."""

import platform

import pytest
from typer.testing import CliRunner

from lightspeed_cli import __version__
from lightspeed_cli.main import app

runner = CliRunner()


@pytest.mark.parametrize("args", [["version"], ["--version"]])
def test_version_information(args: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEFAULT_MODEL", "llama3.1")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-very-secret")

    result = runner.invoke(app, args)

    assert result.exit_code == 0
    assert __version__ in result.output
    assert platform.python_version() in result.output
    assert "llama3.1" in result.output
    assert "http://localhost:11434/v1" in result.output
    assert "sk-very-secret" not in result.output  # never show the key


def test_version_with_default_server() -> None:
    result = runner.invoke(app, ["version"])

    assert "OpenAI (default)" in result.output


def test_version_works_with_invalid_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "TRACE")

    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert __version__ in result.output
    assert "invalid (1 problem(s))" in result.output


def test_no_arguments_shows_help() -> None:
    result = runner.invoke(app, [])

    assert "ask" in result.output
    assert "chat" in result.output
