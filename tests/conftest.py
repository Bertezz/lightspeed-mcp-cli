"""Shared test fixtures."""

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
    }
)


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """Hide the developer's environment and ``.env`` from every test.

    Yields an empty working directory in which tests may write a ``.env`` file.
    """
    for name in list(os.environ):
        if name.upper() in _SETTINGS_ENV_VARS:
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()
