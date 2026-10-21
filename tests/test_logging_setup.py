"""Tests for ``lightspeed_cli.logging_setup``."""

import logging
from io import StringIO

import pytest
from rich.console import Console

from lightspeed_cli.logging_setup import APP_LOGGER, QUIET_LOGGERS, configure_logging


def test_level_comes_from_settings() -> None:
    configure_logging("WARNING")

    assert logging.getLogger(APP_LOGGER).level == logging.WARNING


def test_debug_overrides_the_level() -> None:
    configure_logging("ERROR", debug=True)

    assert logging.getLogger(APP_LOGGER).level == logging.DEBUG


@pytest.mark.parametrize("debug", [False, True])
def test_http_libraries_stay_quiet(debug: bool) -> None:
    # Their DEBUG output can contain the Authorization header (the API key).
    configure_logging("DEBUG", debug=debug)

    for name in QUIET_LOGGERS:
        assert logging.getLogger(name).level == logging.WARNING


def test_calling_twice_does_not_duplicate_output() -> None:
    buffer = StringIO()
    console = Console(file=buffer, width=120)

    configure_logging("INFO", console=console)
    configure_logging("INFO", console=console)
    logging.getLogger(f"{APP_LOGGER}.client").info("hello once")

    assert buffer.getvalue().count("hello once") == 1
    assert len(logging.getLogger(APP_LOGGER).handlers) == 1
