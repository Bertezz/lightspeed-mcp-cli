"""Built-in tools that report on the local machine: CPU, memory, disk and OS.

Learning notes
--------------
This file shows what *writing a tool* looks like. Each tool is a plain Python
function with:

* ``@registry.register()`` on top, which adds it to the registry;
* type hints on every parameter, which become the JSON Schema the AI sees;
* a docstring, whose first paragraph becomes the description the AI reads to
  decide when to use the tool;
* a JSON-friendly return value (dicts, lists, numbers, strings).

``psutil`` ("process and system utilities") reads CPU, memory and disk
information on Linux, macOS and Windows. ``platform`` (standard library)
describes the operating system and Python itself.

Three ideas worth noticing in this file:

1. **TypedDict.** The return values are dicts, but each has a ``TypedDict``
   class describing its exact keys and value types. mypy then catches typos
   like ``"precent"``, and the classes double as documentation of the output.
2. **Partial results.** ``get_system_metrics`` collects each section (cpu,
   memory, ...) separately. If one fails, for example because a locked-down
   container hides a file in ``/proc``, that section becomes ``None`` and the
   reason goes into ``warnings``. The other sections are still returned.
3. **Containers lie (a little).** Inside Docker or OpenShift, psutil reports
   the *host's* RAM and cores, not the container's limits. The real limits live
   in Linux *cgroup* files, which ``_container_limits`` reads.
"""

import os
import platform
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, TypedDict, TypeVar

import psutil
from pydantic import Field

from lightspeed_cli.tools.base import registry

_BYTES_PER_GIB = 1024**3  # 1 GiB = 1024 * 1024 * 1024 bytes

# The root of the filesystem: "/" on Linux and macOS, "C:\\" on Windows.
# Hard-coding "/" would break on Windows.
ROOT_PATH = os.path.abspath(os.sep)

# Where Linux exposes cgroup (control group) information. cgroups are the kernel
# feature containers use to limit how much CPU and memory a process may use.
CGROUP_ROOT = Path("/sys/fs/cgroup")
SELF_CGROUP_FILE = Path("/proc/self/cgroup")

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Output shapes
# --------------------------------------------------------------------------- #
class CpuInfo(TypedDict):
    percent: float
    logical_cores: int
    physical_cores: int


class MemoryInfo(TypedDict):
    total_gib: float
    used_gib: float
    available_gib: float
    percent: float


class DiskInfo(TypedDict):
    path: str
    total_gib: float
    used_gib: float
    free_gib: float
    percent: float


class OsInfo(TypedDict):
    system: str
    release: str
    distribution: str
    architecture: str
    python_version: str


class ContainerLimits(TypedDict):
    # None means "no limit found" (not in a container, or unlimited).
    memory_limit_gib: float | None
    cpu_limit_cores: float | None


class SystemMetrics(TypedDict):
    # Each section is None if it could not be read; see ``warnings``.
    cpu: CpuInfo | None
    memory: MemoryInfo | None
    disk: DiskInfo | None
    os: OsInfo | None
    container_limits: ContainerLimits | None
    warnings: list[str]


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
@registry.register(timeout=10)
def get_system_metrics(
    cpu_interval: Annotated[
        float,
        Field(ge=0.1, le=2, description="Seconds to measure CPU activity over."),
    ] = 0.5,
) -> SystemMetrics:
    """Return a snapshot of CPU, memory, root disk usage, operating system details
    and container resource limits.
    """
    warnings: list[str] = []
    # ``lambda: ...`` creates a tiny function without a name. We pass functions
    # (not their results) to _collect, so that _collect can run each one inside
    # its own try/except. A dict literal is evaluated top to bottom, so the
    # sections run in this order.
    return {
        "cpu": _collect("cpu", lambda: _cpu(cpu_interval), warnings),
        "memory": _collect("memory", _memory, warnings),
        "disk": _collect("disk", lambda: _disk(ROOT_PATH), warnings),
        "os": _collect("os", _os_info, warnings),
        "container_limits": _collect("container_limits", _container_limits, warnings),
        "warnings": warnings,
    }


@registry.register()
def get_disk_usage(
    path: Annotated[
        str, Field(description="Any path on the disk to inspect, e.g. '/home'.")
    ] = "/",
) -> DiskInfo:
    """Return total, used and free space of the disk that contains a given path.

    get_system_metrics already reports the root disk; use this for other disks.
    """
    # If the path does not exist, psutil raises FileNotFoundError. We do not
    # catch it: the registry turns it into an error result for the AI, e.g.
    # "FileNotFoundError: [Errno 2] No such file or directory: '/nope'".
    return _disk(path)


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def _collect(section: str, read: Callable[[], T], warnings: list[str]) -> T | None:
    """Run ``read()``; on an *environment* problem, record a warning instead.

    Only ``OSError`` (files we may not read, missing /proc entries, ...) and
    ``psutil.Error`` (e.g. AccessDenied) are caught. A bug in our own code, such
    as a TypeError, still fails loudly: hiding bugs as "warnings" would make
    them very hard to find.
    """
    try:
        return read()
    except (OSError, psutil.Error) as exc:
        # Some exceptions have no message (str(exc) == ""); then the type name
        # alone is clearer than "AccessDenied: " with a dangling colon.
        reason = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        warnings.append(f"{section}: {reason}")
        return None


def _gib(num_bytes: int) -> float:
    """Convert bytes to GiB, rounded to 2 decimals for readability."""
    return round(num_bytes / _BYTES_PER_GIB, 2)


def _cpu(interval: float) -> CpuInfo:
    # cpu_percent(interval=...) *blocks*: it sleeps for `interval` seconds and
    # measures activity meanwhile. That is why the registry runs sync tools in a
    # worker thread instead of on the event loop.
    return {
        "percent": psutil.cpu_percent(interval=interval),
        # cpu_count() can return None on unusual platforms, hence ``or 0``.
        "logical_cores": psutil.cpu_count(logical=True) or 0,
        "physical_cores": psutil.cpu_count(logical=False) or 0,
    }


def _memory() -> MemoryInfo:
    memory = psutil.virtual_memory()
    return {
        "total_gib": _gib(memory.total),
        "used_gib": _gib(memory.used),
        "available_gib": _gib(memory.available),
        "percent": memory.percent,
    }


def _disk(path: str) -> DiskInfo:
    usage = psutil.disk_usage(path)
    return {
        "path": path,
        "total_gib": _gib(usage.total),
        "used_gib": _gib(usage.used),
        "free_gib": _gib(usage.free),
        "percent": usage.percent,
    }


def _os_info() -> OsInfo:
    # Linux distributions describe themselves in /etc/os-release, which
    # platform.freedesktop_os_release() reads. The file is missing on macOS,
    # Windows and in minimal containers, so fall back to the OS name.
    try:
        distribution = platform.freedesktop_os_release().get("PRETTY_NAME", "")
    except OSError:
        distribution = ""
    return {
        "system": platform.system(),  # "Linux", "Darwin" (macOS), "Windows"
        "release": platform.release(),  # kernel / OS version
        "distribution": distribution or platform.system(),
        # Reported as-is: "x86_64"/"aarch64" (Linux), "arm64" (Apple Silicon),
        # "AMD64" (Windows). The AI understands all of them.
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
    }


# --------------------------------------------------------------------------- #
# Container limits (Linux cgroup v2)
# The paths are parameters so tests can point them at fake files in a
# temporary folder: much simpler than mocking ``open()``.
# --------------------------------------------------------------------------- #
def _container_limits(
    cgroup_root: Path = CGROUP_ROOT, self_cgroup_file: Path = SELF_CGROUP_FILE
) -> ContainerLimits:
    """Read this process's memory and CPU limits from cgroup v2 files.

    Missing files are normal (not Linux, old cgroup v1, or simply no limit) and
    give ``None``. Known gap: cgroup v1 (RHEL 7/8 era hosts) is not supported.
    """
    group_dir = cgroup_root / _own_cgroup_path(self_cgroup_file)
    return {
        "memory_limit_gib": _memory_limit_gib(group_dir / "memory.max"),
        "cpu_limit_cores": _cpu_limit_cores(group_dir / "cpu.max"),
    }


def _own_cgroup_path(self_cgroup_file: Path) -> str:
    """Return this process's cgroup v2 path, relative to the cgroup root.

    /proc/self/cgroup contains a line like ``0::/user.slice/session-1.scope``.
    Inside a container it is usually just ``0::/``.
    """
    try:
        lines = self_cgroup_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    for line in lines:
        if line.startswith("0::"):
            return line.removeprefix("0::").lstrip("/")
    return ""  # only cgroup v1 lines: not supported


def _read_optional(path: Path) -> str | None:
    """Return the stripped file content, or None if the file does not exist."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None


def _memory_limit_gib(path: Path) -> float | None:
    # memory.max holds "max" (unlimited) or a number of bytes.
    value = _read_optional(path)
    if value is None or not value.isdigit():
        return None
    return _gib(int(value))


def _cpu_limit_cores(path: Path) -> float | None:
    # cpu.max holds "<quota> <period>" in microseconds, e.g. "50000 100000":
    # the group may use 50 ms of CPU time every 100 ms, i.e. 0.5 cores.
    # A quota of "max" means unlimited.
    value = _read_optional(path)
    if value is None:
        return None
    quota, _, period = value.partition(" ")
    if not quota.isdigit() or not period.isdigit() or int(period) == 0:
        return None
    return round(int(quota) / int(period), 2)
