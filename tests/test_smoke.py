"""Smoke tests for the package scaffold."""

from typer.testing import CliRunner

from lightspeed_cli import __version__
from lightspeed_cli.main import app

runner = CliRunner()


def test_version_is_defined() -> None:
    assert __version__


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert __version__ in result.output
