"""Built-in tools that read information about the local machine.

Learning notes
--------------
This file shows what *writing a tool* looks like. Each tool is a plain Python
function with:

* ``@registry.register()`` on top, which adds it to the registry;
* type hints on every parameter, which become the JSON Schema the AI sees;
* a docstring, whose first paragraph becomes the description the AI reads to
  decide when to use the tool;
* a JSON-friendly return value (dicts, lists, numbers, strings).

``Annotated[float, Field(...)]`` attaches extra information to a type: limits
(``ge`` = greater or equal, ``le`` = less or equal) that are checked before the
tool runs, and a description that is shown to the AI.

``psutil`` ("process and system utilities") is a library that reads CPU, memory
and disk information on Linux, macOS and Windows.

These tools are registered as soon as this module is imported;
``tools/__init__.py`` imports it so that happens automatically.
"""

from typing import Annotated

import psutil
from pydantic import Field

from lightspeed_cli.tools.base import registry

_BYTES_PER_GIB = 1024**3  # 1 GiB = 1024 * 1024 * 1024 bytes


def _gib(num_bytes: int) -> float:
    """Convert bytes to GiB, rounded to 2 decimals for readability."""
    return round(num_bytes / _BYTES_PER_GIB, 2)


# A short timeout on purpose: sampling the CPU takes at most 5 seconds.
@registry.register(timeout=10)
def get_cpu_usage(
    interval: Annotated[
        float,
        Field(ge=0.1, le=5, description="Seconds to measure CPU activity over."),
    ] = 0.5,
) -> dict[str, float | int]:
    """Return the current CPU utilisation percentage and the number of cores."""
    # cpu_percent(interval=...) *blocks*: it sleeps for `interval` seconds and
    # measures activity meanwhile. That is why the registry runs sync tools in a
    # worker thread instead of on the event loop.
    return {
        "percent": psutil.cpu_percent(interval=interval),
        # cpu_count() can return None on unusual platforms, hence ``or 0``.
        "logical_cores": psutil.cpu_count(logical=True) or 0,
        "physical_cores": psutil.cpu_count(logical=False) or 0,
    }


@registry.register()
def get_memory_usage() -> dict[str, float]:
    """Return total, used and available RAM in GiB, plus the percentage used."""
    memory = psutil.virtual_memory()
    return {
        "total_gib": _gib(memory.total),
        "used_gib": _gib(memory.used),
        "available_gib": _gib(memory.available),
        "percent": memory.percent,
    }


@registry.register()
def get_disk_usage(
    path: Annotated[
        str, Field(description="Any path on the disk to inspect, e.g. '/'.")
    ] = "/",
) -> dict[str, float | str]:
    """Return total, used and free space of the disk that contains a path."""
    # If the path does not exist, psutil raises FileNotFoundError. We do not
    # catch it: the registry turns it into an error result for the AI, e.g.
    # "FileNotFoundError: [Errno 2] No such file or directory: '/nope'".
    usage = psutil.disk_usage(path)
    return {
        "path": path,
        "total_gib": _gib(usage.total),
        "used_gib": _gib(usage.used),
        "free_gib": _gib(usage.free),
        "percent": usage.percent,
    }
