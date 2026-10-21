"""Lightspeed MCP command-line interface.

A folder that contains an ``__init__.py`` file is a Python *package*: something
you can ``import``. This file runs the first time anyone writes
``import lightspeed_cli`` (or imports anything inside it).

Keep it tiny: everything here runs on every import, even for ``--help``.
"""

# The single source of truth for the version number. ``pyproject.toml`` tells
# hatchling to read it from here (``[tool.hatch.version]``), and the
# ``lightspeed version`` command prints it. Change it here and only here.
__version__ = "0.1.0"

# ``__all__`` lists the names this module officially "exports". It controls what
# ``from lightspeed_cli import *`` brings in and tells tools what is public API.
__all__ = ["__version__"]
