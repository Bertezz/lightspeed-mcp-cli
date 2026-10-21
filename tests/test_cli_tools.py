"""Tests for the ``lightspeed tools`` commands.

Learning notes
--------------
``CliRunner.invoke(app, [...])`` runs the CLI in-process. ``result.exit_code``
is what a shell would see (0 = success, 1 = our error, 2 = usage error), and
``result.output`` holds the printed text.
"""

import json

from pytest_mock import MockerFixture
from typer.testing import CliRunner

from lightspeed_cli.main import app

runner = CliRunner()


def test_tools_list_shows_builtin_tools() -> None:
    result = runner.invoke(app, ["tools", "list"])

    assert result.exit_code == 0
    for name in ("get_system_metrics", "get_disk_usage", "inspect_git_status"):
        assert name in result.output


def test_tools_list_json_is_the_openai_format() -> None:
    result = runner.invoke(app, ["tools", "list", "--json"])

    assert result.exit_code == 0
    tools = json.loads(result.output)
    assert {tool["function"]["name"] for tool in tools} >= {"get_system_metrics"}
    assert all(tool["type"] == "function" for tool in tools)


def test_tools_run_prints_json_output(mocker: MockerFixture) -> None:
    mocker.patch(
        "lightspeed_cli.tools.system_tools.psutil.cpu_percent", return_value=42.0
    )

    result = runner.invoke(
        app, ["tools", "run", "get_system_metrics", '{"cpu_interval": 0.1}']
    )

    assert result.exit_code == 0
    assert json.loads(result.output)["cpu"]["percent"] == 42.0


def test_tools_run_inspect_git_status_outside_a_repository() -> None:
    # conftest.py runs every test in an empty temporary folder: not a repo.
    result = runner.invoke(app, ["tools", "run", "inspect_git_status"])

    assert result.exit_code == 0
    assert json.loads(result.output)["is_git_repo"] is False


def test_tools_run_reports_tool_errors() -> None:
    result = runner.invoke(
        app, ["tools", "run", "get_disk_usage", '{"path": "/does/not/exist"}']
    )

    assert result.exit_code == 1
    assert "FileNotFoundError" in result.output


def test_tools_run_reports_unknown_tool() -> None:
    result = runner.invoke(app, ["tools", "run", "no_such_tool"])

    assert result.exit_code == 1
    assert "Unknown tool" in result.output


def test_tools_run_rejects_invalid_json() -> None:
    result = runner.invoke(app, ["tools", "run", "get_system_metrics", "{not json"])

    assert result.exit_code == 2
    assert "not valid JSON" in result.output
