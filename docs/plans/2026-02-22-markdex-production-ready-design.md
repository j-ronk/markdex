# Markdex Production-Ready Design

## Goal

Convert markdex from a flat collection of scripts into an installable Python package with a proper CLI, centralized config, and an MCP tool for in-conversation indexing. Target audience: teammates installing from a git repo.

## Project Structure

```
markdex/
├── pyproject.toml
├── README.md
├── src/
│   └── markdex/
│       ├── __init__.py
│       ├── cli.py          # argparse CLI: index, serve, list
│       ├── chunker.py      # existing, unchanged
│       ├── config.py       # paths, constants, defaults
│       ├── indexer.py       # refactored to use config
│       └── server.py        # refactored to use config + new index_document tool
├── tests/
│   ├── test_chunker.py
│   ├── test_indexer.py
│   └── test_server.py
└── docs/                    # user content (markdown files to index)
```

## Configuration & Data Storage

All data lives in `~/.markdex/`:

```
~/.markdex/
├── chroma_db/
└── config.toml         # optional overrides (future use)
```

`config.py` centralizes all defaults:
- `DATA_DIR = ~/.markdex/`
- `CHROMA_DB_PATH = ~/.markdex/chroma_db`
- `COLLECTION_NAME = "markdown_docs"`
- `EMBEDDING_MODEL = "all-mpnet-base-v2"`

Data directory is created automatically on first use. Eliminates the constant duplication between indexer.py and server.py.

## CLI Design

Entry point: `markdex` (via `[project.scripts]` in pyproject.toml)

```
markdex index <path>           # index a file or directory of .md files
markdex index <path> --reset   # clear index first, then index
markdex serve                  # start MCP server (stdio transport)
markdex list                   # list indexed files
```

`markdex index` accepts a single .md file or a directory (recursive). `markdex serve` replaces `python server.py`. `markdex list` is a convenience to check indexed content without starting Claude.

## MCP Server Changes

New tool added:

```
index_document(path: str) -> str
```

- Accepts absolute path to a .md file or directory
- Runs indexer in-process (reuses loaded embedding model)
- Returns summary: "Indexed 45 chunks from 1 file(s)"
- No server restart needed — ChromaDB collection is shared

Existing tools (`query_docs`, `list_indexed_files`) lose their `chroma_path` and `collection_name` parameters. Those are now handled by config.py.

## Installation

```bash
git clone <repo-url>
cd markdex
pip install -e .
```

MCP setup (manual, documented in README):

```bash
claude mcp add --scope user --transport stdio markdex -- markdex serve
```

## What Changes

- Source files move into `src/markdex/`
- `requirements.txt` replaced by `pyproject.toml`
- Hardcoded relative paths replaced by `~/.markdex/`
- Constants centralized in `config.py`
- CLI entry point replaces `python <script>.py` invocations
- New `index_document` MCP tool for in-conversation indexing
- README and CLAUDE.md updated
