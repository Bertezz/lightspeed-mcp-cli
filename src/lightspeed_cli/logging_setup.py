"""Configure Python's ``logging`` for the CLI.

Learning notes
--------------
Library code (client.py, tools/base.py, ...) only *creates* log messages, with
``logging.getLogger(__name__).debug(...)``. Deciding where messages go and
which ones are shown is the job of the *application*, once, at start-up; that
is this module.

Loggers form a tree by name: ``lightspeed_cli.client`` is a child of
``lightspeed_cli``. Setting the level on ``lightspeed_cli`` therefore controls
all of our modules at once, without touching other libraries' loggers.

Security: we deliberately keep the ``openai`` and ``httpx2`` loggers at
WARNING, even in ``--debug`` mode. Their DEBUG output describes raw HTTP
requests, which can include the Authorization header, i.e. your API key.
"""

import logging

from rich.console import Console
from rich.logging import RichHandler

APP_LOGGER = "lightspeed_cli"

# Third-party loggers that must never go below WARNING (see the module notes).
QUIET_LOGGERS = ("openai", "httpx2", "httpx", "httpcore")


def configure_logging(
    level: str, *, debug: bool = False, console: Console | None = None
) -> None:
    """Send our log messages to stderr, at ``level`` (or DEBUG with ``debug``).

    Safe to call more than once: earlier handlers are replaced, not duplicated.
    """
    logger = logging.getLogger(APP_LOGGER)
    logger.setLevel(logging.DEBUG if debug else level)
    # Remove handlers from a previous call, so messages are not printed twice.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    # RichHandler formats log records nicely (time, level colour, message).
    logger.addHandler(
        RichHandler(console=console or Console(stderr=True), show_path=False)
    )
    # Do not also pass our records up to the root logger (which might print
    # them a second time if some other code configured it).
    logger.propagate = False

    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
