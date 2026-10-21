"""Tools the Lightspeed assistant can call, and the registry that manages them.

Other code should import from here (``from lightspeed_cli.tools import
registry``) rather than from the submodules. That way the internal file layout
can change without breaking imports elsewhere.
"""

# Importing ``system`` runs its ``@registry.register()`` decorators, which adds
# the built-in tools to the default registry. We import it for that *side
# effect* only, so the linter's "imported but unused" warning (F401) is
# silenced on purpose.
from lightspeed_cli.tools import system  # noqa: F401
from lightspeed_cli.tools.base import (
    DEFAULT_TIMEOUT,
    RegisteredTool,
    ToolRegistry,
    registry,
)

__all__ = ["DEFAULT_TIMEOUT", "RegisteredTool", "ToolRegistry", "registry"]
