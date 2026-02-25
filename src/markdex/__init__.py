"""Markdex: local Markdown indexing and retrieval via MCP."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("markdex")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"
