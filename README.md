# markdex

Local Markdown indexing and retrieval via MCP. Index your Markdown documents and query them from Claude Code using semantic search.

## Install

```bash
git clone <repo-url>
cd markdex
pipx install -e . --python python3.13
```

This puts `markdex` on your PATH globally via an isolated venv. Requires [pipx](https://pipx.pypa.io/) (`brew install pipx`). Use Python 3.13 or earlier (chromadb doesn't yet support 3.14).

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

## Commands

| Command | Description |
|---------|-------------|
| `markdex index <path> [--reset]` | Index .md files from a file or directory |
| `markdex serve` | Start the MCP server (stdio transport) |
| `markdex list` | List all indexed files |

## Data Storage

All data is stored in `~/.markdex/`:
- `chroma_db/` — vector store (ChromaDB)

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
