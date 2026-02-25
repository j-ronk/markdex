# markdex

Index your Markdown files and make them retrievable by AI through semantic search.

markdex chunks your documents by heading, embeds them locally, and serves them over [MCP](https://modelcontextprotocol.io/) so tools like Claude Code can find exactly the right section when they need it. After initial model download, everything runs locally — your documents never leave your machine.

## Install

```bash
pip install markdex
```

Or install globally with [pipx](https://pipx.pypa.io/):

```bash
pipx install markdex
```

Requires Python 3.10–3.13 (chromadb doesn't yet support 3.14).

## Quick Start

Index your documents:

```bash
markdex index path/to/docs/
```

Check what's indexed:

```bash
markdex list
```

## Claude Code Integration

Add markdex as a global MCP server:

```bash
claude mcp add --scope user --transport stdio markdex -- markdex serve
```

Then in any Claude Code session, you can:
- **Search** your indexed docs (the `query_docs` tool)
- **List** indexed files (the `list_indexed_files` tool)
- **Index** new documents from conversation (the `index_document` tool)
- **Remove** documents from the index (the `remove_document` tool)

## Commands

| Command | Description |
|---------|-------------|
| `markdex index <path> [--reset]` | Index .md files from a file or directory |
| `markdex serve` | Start the MCP server (stdio transport) |
| `markdex list` | List all indexed files |
| `markdex remove <number>` | Remove a file from the index (by number from `list`) |

## Data Storage

All data is stored in `~/.markdex/`:
- `chroma_db/` — vector store (ChromaDB)

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
