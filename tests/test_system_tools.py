"""Tests for the built-in system tools (``lightspeed_cli.tools.system_tools``).

Learning notes
--------------
Real CPU and memory numbers change all the time, so a test cannot check exact
values. Instead we *mock* psutil: ``mocker.patch(...)`` (from pytest-mock)
temporarily replaces a function with a fake that returns whatever we choose,
and undoes the replacement after the test. Then the result is predictable.

Mock the name *where it is used*: ``lightspeed_cli.tools.system_tools.psutil``,
because that is the psutil reference our code looks up.

The container-limit tests do not mock anything: they write fake cgroup files
into a temporary folder and point the function at it.
"""

from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest
from pytest_mock import MockerFixture

from lightspeed_cli.tools import registry
from lightspeed_cli.tools.system_tools import (
    ROOT_PATH,
    _container_limits,
    get_disk_usage,
    get_system_metrics,
)

GIB = 1024**3
MODULE = "lightspeed_cli.tools.system_tools"


@pytest.fixture
def fake_machine(mocker: MockerFixture) -> None:
    """Replace every psutil/platform call with predictable fakes."""
    mocker.patch(f"{MODULE}.psutil.cpu_percent", return_value=12.5)
    # side_effect with a function: the fake computes its answer from the
    # arguments, like the real cpu_count(logical=...) does.
    mocker.patch(
        f"{MODULE}.psutil.cpu_count",
        side_effect=lambda logical: 8 if logical else 4,
    )
    # SimpleNamespace is a quick object with attributes: a stand-in for the
    # named tuples psutil really returns.
    mocker.patch(
        f"{MODULE}.psutil.virtual_memory",
        return_value=SimpleNamespace(
            total=16 * GIB, used=6 * GIB, available=10 * GIB, percent=37.5
        ),
    )
    mocker.patch(
        f"{MODULE}.psutil.disk_usage",
        return_value=SimpleNamespace(
            total=500 * GIB, used=125 * GIB, free=375 * GIB, percent=25.0
        ),
    )
    mocker.patch(f"{MODULE}.platform.system", return_value="Linux")
    mocker.patch(f"{MODULE}.platform.release", return_value="6.6.0")
    mocker.patch(f"{MODULE}.platform.machine", return_value="aarch64")
    mocker.patch(f"{MODULE}.platform.python_version", return_value="3.12.3")
    mocker.patch(
        f"{MODULE}.platform.freedesktop_os_release",
        return_value={"PRETTY_NAME": "Red Hat Enterprise Linux 9.4 (Plow)"},
    )
    mocker.patch(
        f"{MODULE}._container_limits",
        return_value={"memory_limit_gib": 0.5, "cpu_limit_cores": 2.0},
    )


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
def test_builtin_tools_are_registered() -> None:
    # An exact set: adding or removing a tool by accident fails this test.
    assert set(registry.names()) == {
        "get_system_metrics",
        "get_disk_usage",
        "inspect_git_status",
    }


def test_cpu_interval_limits_are_in_the_schema() -> None:
    tool = registry.get("get_system_metrics")

    assert tool is not None
    assert tool.definition.parameters.properties["cpu_interval"] == {
        "type": "number",
        "minimum": 0.1,
        "maximum": 2,
        "default": 0.5,
        "description": "Seconds to measure CPU activity over.",
    }


# --------------------------------------------------------------------------- #
# get_system_metrics
# --------------------------------------------------------------------------- #
@pytest.mark.usefixtures("fake_machine")
def test_full_snapshot() -> None:
    assert get_system_metrics(cpu_interval=0.2) == {
        "cpu": {"percent": 12.5, "logical_cores": 8, "physical_cores": 4},
        "memory": {
            "total_gib": 16.0,
            "used_gib": 6.0,
            "available_gib": 10.0,
            "percent": 37.5,
        },
        "disk": {
            "path": ROOT_PATH,
            "total_gib": 500.0,
            "used_gib": 125.0,
            "free_gib": 375.0,
            "percent": 25.0,
        },
        "os": {
            "system": "Linux",
            "release": "6.6.0",
            "distribution": "Red Hat Enterprise Linux 9.4 (Plow)",
            "architecture": "aarch64",
            "python_version": "3.12.3",
        },
        "container_limits": {"memory_limit_gib": 0.5, "cpu_limit_cores": 2.0},
        "warnings": [],
    }


@pytest.mark.usefixtures("fake_machine")
def test_cpu_interval_is_passed_to_psutil(mocker: MockerFixture) -> None:
    cpu_percent = mocker.patch(f"{MODULE}.psutil.cpu_percent", return_value=1.0)

    get_system_metrics(cpu_interval=1.5)

    # We can also check *how* a fake was called.
    cpu_percent.assert_called_once_with(interval=1.5)


@pytest.mark.usefixtures("fake_machine")
def test_unknown_core_count_becomes_zero(mocker: MockerFixture) -> None:
    mocker.patch(f"{MODULE}.psutil.cpu_count", return_value=None)

    cpu = get_system_metrics()["cpu"]

    assert cpu is not None
    assert (cpu["logical_cores"], cpu["physical_cores"]) == (0, 0)


@pytest.mark.usefixtures("fake_machine")
def test_unreadable_section_gives_partial_result(mocker: MockerFixture) -> None:
    mocker.patch(
        f"{MODULE}.psutil.disk_usage", side_effect=PermissionError("access denied")
    )
    mocker.patch(f"{MODULE}.psutil.cpu_percent", side_effect=psutil.AccessDenied())

    metrics = get_system_metrics()

    assert metrics["disk"] is None
    assert metrics["cpu"] is None
    assert metrics["memory"] is not None  # the other sections survived
    assert metrics["os"] is not None
    assert metrics["warnings"] == [
        "cpu: AccessDenied",  # psutil.AccessDenied() has no message
        "disk: PermissionError: access denied",
    ]


@pytest.mark.usefixtures("fake_machine")
def test_bugs_are_not_hidden_as_warnings(mocker: MockerFixture) -> None:
    mocker.patch(f"{MODULE}.psutil.virtual_memory", side_effect=TypeError("bug"))

    # Called through the registry: the bug becomes an *error result*.
    result = registry.execute_sync("get_system_metrics", {"cpu_interval": 0.1})

    assert result.error == "TypeError: bug"


@pytest.mark.usefixtures("fake_machine")
def test_missing_os_release_falls_back_to_system_name(mocker: MockerFixture) -> None:
    mocker.patch(
        f"{MODULE}.platform.freedesktop_os_release", side_effect=OSError("no file")
    )

    os_info = get_system_metrics()["os"]

    assert os_info is not None
    assert os_info["distribution"] == "Linux"


def test_real_snapshot_through_the_registry() -> None:
    # No mocking: proves the tool works on the machine running the tests.
    result = registry.execute_sync("get_system_metrics", {"cpu_interval": 0.1})

    assert result.status == "success"
    assert isinstance(result.output, dict)
    assert set(result.output) == {
        "cpu",
        "memory",
        "disk",
        "os",
        "container_limits",
        "warnings",
    }


# --------------------------------------------------------------------------- #
# Container limits (fake cgroup files in tmp_path)
# --------------------------------------------------------------------------- #
def write_cgroup(
    tmp_path: Path, own_path: str, memory_max: str | None, cpu_max: str | None
) -> tuple[Path, Path]:
    """Create a fake /sys/fs/cgroup tree and a fake /proc/self/cgroup file."""
    root = tmp_path / "cgroup"
    group = root / own_path.lstrip("/")
    group.mkdir(parents=True)
    if memory_max is not None:
        (group / "memory.max").write_text(f"{memory_max}\n", encoding="utf-8")
    if cpu_max is not None:
        (group / "cpu.max").write_text(f"{cpu_max}\n", encoding="utf-8")
    self_cgroup = tmp_path / "self_cgroup"
    self_cgroup.write_text(f"0::{own_path}\n", encoding="utf-8")
    return root, self_cgroup


@pytest.mark.parametrize(
    ("memory_max", "cpu_max", "expected"),
    [
        (
            "536870912",
            "50000 100000",
            {"memory_limit_gib": 0.5, "cpu_limit_cores": 0.5},
        ),
        ("max", "max 100000", {"memory_limit_gib": None, "cpu_limit_cores": None}),
        (None, None, {"memory_limit_gib": None, "cpu_limit_cores": None}),
        ("garbage", "100 0", {"memory_limit_gib": None, "cpu_limit_cores": None}),
    ],
    ids=["limited", "unlimited", "files-missing", "malformed"],
)
def test_container_limits(
    tmp_path: Path,
    memory_max: str | None,
    cpu_max: str | None,
    expected: dict[str, float | None],
) -> None:
    root, self_cgroup = write_cgroup(tmp_path, "/kubepods/pod1", memory_max, cpu_max)

    assert _container_limits(root, self_cgroup) == expected


def test_container_root_cgroup(tmp_path: Path) -> None:
    # Inside a container, /proc/self/cgroup usually says just "0::/".
    root, self_cgroup = write_cgroup(tmp_path, "/", "1073741824", "200000 100000")

    assert _container_limits(root, self_cgroup) == {
        "memory_limit_gib": 1.0,
        "cpu_limit_cores": 2.0,
    }


def test_container_limits_without_cgroup_v2(tmp_path: Path) -> None:
    root = tmp_path / "cgroup"
    root.mkdir()
    v1_only = tmp_path / "self_cgroup"
    v1_only.write_text("12:memory:/docker/abc\n", encoding="utf-8")

    assert _container_limits(root, v1_only) == {
        "memory_limit_gib": None,
        "cpu_limit_cores": None,
    }
    # A missing /proc/self/cgroup (e.g. macOS) gives the same answer.
    assert _container_limits(root, tmp_path / "missing") == {
        "memory_limit_gib": None,
        "cpu_limit_cores": None,
    }


# --------------------------------------------------------------------------- #
# get_disk_usage
# --------------------------------------------------------------------------- #
@pytest.mark.usefixtures("fake_machine")
def test_get_disk_usage(mocker: MockerFixture) -> None:
    disk_usage = mocker.patch(
        f"{MODULE}.psutil.disk_usage",
        return_value=SimpleNamespace(
            total=100 * GIB, used=40 * GIB, free=60 * GIB, percent=40.0
        ),
    )

    assert get_disk_usage("/home") == {
        "path": "/home",
        "total_gib": 100.0,
        "used_gib": 40.0,
        "free_gib": 60.0,
        "percent": 40.0,
    }
    disk_usage.assert_called_once_with("/home")


def test_missing_path_becomes_error_result() -> None:
    result = registry.execute_sync("get_disk_usage", {"path": "/does/not/exist"})

    assert result.status == "error"
    assert result.error is not None
    assert result.error.startswith("FileNotFoundError")
