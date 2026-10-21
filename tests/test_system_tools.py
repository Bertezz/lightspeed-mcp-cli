"""Tests for the built-in system tools (``lightspeed_cli.tools.system``).

Learning notes
--------------
Real CPU and memory numbers change all the time, so a test cannot check exact
values. Instead we *mock* psutil: ``mocker.patch(...)`` (from pytest-mock)
temporarily replaces a function with a fake that returns whatever we choose,
and undoes the replacement after the test. Then the result is predictable.

Mock the name *where it is used*: ``lightspeed_cli.tools.system.psutil...``,
because that is the psutil reference our code looks up.
"""

from types import SimpleNamespace

from pytest_mock import MockerFixture

from lightspeed_cli.tools import registry
from lightspeed_cli.tools.system import (
    get_cpu_usage,
    get_disk_usage,
    get_memory_usage,
)

GIB = 1024**3


def test_builtin_tools_are_registered() -> None:
    assert {"get_cpu_usage", "get_memory_usage", "get_disk_usage"} <= set(
        registry.names()
    )


def test_cpu_interval_limits_are_in_the_schema() -> None:
    tool = registry.get("get_cpu_usage")

    assert tool is not None
    assert tool.definition.parameters.properties["interval"] == {
        "type": "number",
        "minimum": 0.1,
        "maximum": 5,
        "default": 0.5,
        "description": "Seconds to measure CPU activity over.",
    }


def test_get_cpu_usage(mocker: MockerFixture) -> None:
    cpu_percent = mocker.patch(
        "lightspeed_cli.tools.system.psutil.cpu_percent", return_value=12.5
    )
    # side_effect with a function: the fake computes its answer from the
    # arguments, like the real cpu_count(logical=...) does.
    mocker.patch(
        "lightspeed_cli.tools.system.psutil.cpu_count",
        side_effect=lambda logical: 8 if logical else 4,
    )

    assert get_cpu_usage(interval=1.0) == {
        "percent": 12.5,
        "logical_cores": 8,
        "physical_cores": 4,
    }
    # We can also check *how* the fake was called.
    cpu_percent.assert_called_once_with(interval=1.0)


def test_get_cpu_usage_handles_unknown_core_count(mocker: MockerFixture) -> None:
    mocker.patch("lightspeed_cli.tools.system.psutil.cpu_percent", return_value=1.0)
    mocker.patch("lightspeed_cli.tools.system.psutil.cpu_count", return_value=None)

    result = get_cpu_usage()

    assert result["logical_cores"] == 0
    assert result["physical_cores"] == 0


def test_get_memory_usage(mocker: MockerFixture) -> None:
    # SimpleNamespace is a quick object with attributes: a stand-in for the
    # named tuple psutil.virtual_memory() really returns.
    fake = SimpleNamespace(
        total=16 * GIB, used=6 * GIB, available=10 * GIB, percent=37.5
    )
    mocker.patch("lightspeed_cli.tools.system.psutil.virtual_memory", return_value=fake)

    assert get_memory_usage() == {
        "total_gib": 16.0,
        "used_gib": 6.0,
        "available_gib": 10.0,
        "percent": 37.5,
    }


def test_get_disk_usage(mocker: MockerFixture) -> None:
    fake = SimpleNamespace(
        total=500 * GIB, used=125 * GIB, free=375 * GIB, percent=25.0
    )
    disk_usage = mocker.patch(
        "lightspeed_cli.tools.system.psutil.disk_usage", return_value=fake
    )

    assert get_disk_usage("/home") == {
        "path": "/home",
        "total_gib": 500.0,
        "used_gib": 125.0,
        "free_gib": 375.0,
        "percent": 25.0,
    }
    disk_usage.assert_called_once_with("/home")


def test_missing_path_becomes_error_result() -> None:
    # No mocking: a real call through the registry with a path that does not
    # exist shows the error containment end to end.
    result = registry.execute_sync("get_disk_usage", {"path": "/does/not/exist"})

    assert result.status == "error"
    assert result.error is not None
    assert result.error.startswith("FileNotFoundError")


def test_real_tools_return_json_through_the_registry() -> None:
    # Smallest allowed interval keeps the test fast.
    result = registry.execute_sync("get_cpu_usage", {"interval": 0.1})

    assert result.status == "success"
    assert isinstance(result.output, dict)
    assert set(result.output) == {"percent", "logical_cores", "physical_cores"}
