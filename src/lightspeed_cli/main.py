"""CLI entry point for ``lightspeed``."""

import typer
from rich.console import Console

from lightspeed_cli import __version__

app = typer.Typer(
    name="lightspeed",
    help="Lightspeed MCP command-line interface.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def main() -> None:
    """Lightspeed MCP command-line interface."""


@app.command()
def version() -> None:
    """Show the installed version."""
    console.print(f"lightspeed-mcp-cli {__version__}")
