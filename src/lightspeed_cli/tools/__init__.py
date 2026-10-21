"""Tools the Lightspeed assistant can call, and the registry that manages them.

Other code should import from here (``from lightspeed_cli.tools import
registry``) rather than from the submodules. That way the internal file layout
can change without breaking imports elsewhere.
"""

# Importing a tool module runs its ``@registry.register()`` decorators, which
# adds its tools to the default registry. We import these modules for that
# *side effect* only, so the linter's "imported but unused" warning (F401) is
# silenced on purpose. To add a new tool module, import it here.
from lightspeed_cli.tools import git_tools, system_tools  # noqa: F401
from lightspeed_cli.tools.base import (
    DEFAULT_TIMEOUT,
    RegisteredTool,
    ToolRegistry,
    registry,
)

__all__ = ["DEFAULT_TIMEOUT", "RegisteredTool", "ToolRegistry", "registry"]
