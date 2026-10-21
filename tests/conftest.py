"""Shared test fixtures.

Learning notes
--------------
pytest automatically loads ``conftest.py`` before running the tests in this
folder. Fixtures defined here are available to every test file.

A *fixture* is a function that prepares something a test needs. A test asks for
a fixture simply by having a parameter with the fixture's name; pytest calls the
fixture and passes the result in. ``monkeypatch`` and ``tmp_path`` used below
are fixtures that pytest itself provides.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from lightspeed_cli.config import get_settings

_SETTINGS_ENV_VARS = frozenset(
    {
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "DEFAULT_MODEL",
        "LOG_LEVEL",
        "REQUEST_TIMEOUT",
        "LIGHTSPEED_DEFAULT_MODEL",
        "LIGHTSPEED_LOG_LEVEL",
        "LIGHTSPEED_REQUEST_TIMEOUT",
        "MAX_TOOL_ITERATIONS",
        "LIGHTSPEED_MAX_TOOL_ITERATIONS",
        "LLM_MAX_RETRIES",
        "LIGHTSPEED_LLM_MAX_RETRIES",
    }
)


# autouse=True -> this fixture runs around *every* test, even tests that do not
# ask for it. Without it, a test could pass on your laptop but fail on a
# colleague's machine, just because their environment or .env file differs.
@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """Hide the developer's environment and ``.env`` from every test.

    Yields an empty working directory in which tests may write a ``.env`` file.
    """
    # monkeypatch changes things temporarily and undoes them after the test.
    for name in list(os.environ):
        if name.upper() in _SETTINGS_ENV_VARS:
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)  # an empty temp folder: no real .env here
    get_settings.cache_clear()
    # Everything before ``yield`` is setup; the test runs at the ``yield``;
    # everything after it is cleanup (it runs even if the test fails).
    yield tmp_path
    get_settings.cache_clear()
