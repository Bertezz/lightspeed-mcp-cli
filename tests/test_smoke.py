"""Smoke tests for the package scaffold.

Learning notes
--------------
A "smoke test" checks the very basics: does the program even start? (The name
comes from hardware: switch it on and see whether smoke comes out.)

pytest finds every function whose name starts with ``test_`` and runs it. Inside,
``assert <condition>`` means "this must be true, otherwise the test fails".
"""

from typer.testing import CliRunner

from lightspeed_cli import __version__
from lightspeed_cli.main import app

# CliRunner runs the CLI inside the test, as if you typed the command in a
# terminal, and captures what it prints so we can check it.
runner = CliRunner()


def test_version_is_defined() -> None:
    assert __version__


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert __version__ in result.output
