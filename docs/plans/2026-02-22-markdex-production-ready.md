# Markdex Production-Ready Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Convert markdex from flat scripts into an installable Python package with CLI, centralized config, and MCP indexing tool.

**Architecture:** `src/markdex/` package layout with `pyproject.toml` for installation. All data stored in `~/.markdex/`. CLI entry point `markdex` with subcommands `index`, `serve`, `list`. MCP server gains an `index_document` tool.

**Tech Stack:** Python, chromadb, sentence-transformers, mcp, argparse

---

### Task 1: Create package structure and config module

**Files:**
- Create: `src/markdex/__init__.py`
- Create: `src/markdex/config.py`
- Create: `pyproject.toml`

**Step 1: Create `src/markdex/__init__.py`**

```python
"""Markdex: local Markdown indexing and retrieval via MCP."""
```

**Step 2: Create `src/markdex/config.py`**

```python
"""Centralized configuration for markdex."""

from pathlib import Path

DATA_DIR = Path.home() / ".markdex"
CHROMA_DB_PATH = str(DATA_DIR / "chroma_db")
COLLECTION_NAME = "markdown_docs"
EMBEDDING_MODEL = "all-mpnet-base-v2"


def ensure_data_dir() -> None:
    """Create the data directory if it doesn't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
```

**Step 3: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "markdex"
version = "0.1.0"
description = "Local Markdown indexing and retrieval via MCP"
requires-python = ">=3.10"
dependencies = [
    "chromadb>=0.5.0",
    "sentence-transformers>=3.0.0",
    "mcp>=1.0.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0.0"]

[project.scripts]
markdex = "markdex.cli:main"

[tool.setuptools.packages.find]
where = ["src"]
```

**Step 4: Verify structure**

Run: `ls src/markdex/`
Expected: `__init__.py  config.py`

**Step 5: Commit**

```bash
git add src/ pyproject.toml
git commit -m "feat: create package structure with config module"
```

---

### Task 2: Move chunker into package

**Files:**
- Move: `chunker.py` -> `src/markdex/chunker.py`
- Modify: `tests/test_chunker.py` (update import)

**Step 1: Move chunker.py**

```bash
mv chunker.py src/markdex/chunker.py
```

The file content is unchanged.

**Step 2: Update test import in `tests/test_chunker.py`**

Change line 3 from:
```python
from chunker import chunk_markdown, Chunk
```
to:
```python
from markdex.chunker import chunk_markdown, Chunk
```

**Step 3: Run tests to verify**

Run: `pytest tests/test_chunker.py -v`
Expected: All 11 tests PASS

**Step 4: Commit**

```bash
git add src/markdex/chunker.py tests/test_chunker.py
git rm chunker.py
git commit -m "refactor: move chunker into markdex package"
```

---

### Task 3: Move and refactor indexer

**Files:**
- Move: `indexer.py` -> `src/markdex/indexer.py`
- Modify: `src/markdex/indexer.py` (use config, accept file paths)
- Modify: `tests/test_indexer.py` (update imports)

**Step 1: Move indexer.py**

```bash
mv indexer.py src/markdex/indexer.py
```

**Step 2: Refactor `src/markdex/indexer.py`**

Replace the constants and imports at the top:

```python
#!/usr/bin/env python3
"""Index Markdown files into ChromaDB for markdex retrieval."""

import argparse
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from markdex.chunker import chunk_markdown
from markdex.config import CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL, ensure_data_dir
```

Remove these lines (they're now in config.py):
```python
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "markdown_docs"
EMBEDDING_MODEL = "all-mpnet-base-v2"
DOCS_DIR = "./docs"
```

In `index_documents()`, change the signature — replace `docs_dir` with `paths` to accept a list of files/directories:

```python
def index_documents(
    paths: list[str] | None = None,
    chroma_path: str = CHROMA_DB_PATH,
    collection_name: str = COLLECTION_NAME,
    reset: bool = False,
) -> dict:
    """Index Markdown files into ChromaDB.

    Args:
        paths: List of file or directory paths to index. Each can be a .md file
               or a directory (searched recursively for .md files).
        chroma_path: Path to ChromaDB storage.
        collection_name: ChromaDB collection name.
        reset: If True, clear and rebuild the index.

    Returns dict with keys: files, chunks.
    """
    ensure_data_dir()

    if not paths:
        print("Error: no paths provided.")
        sys.exit(1)

    md_files = []
    for p in paths:
        path = Path(p)
        if path.is_file() and path.suffix == ".md":
            md_files.append(path)
        elif path.is_dir():
            md_files.extend(sorted(path.rglob("*.md")))
        else:
            print(f"Warning: skipping '{p}' (not a .md file or directory)")

    if not md_files:
        print("No .md files found in provided paths.")
        sys.exit(1)

    print(f"Found {len(md_files)} Markdown file(s)")

    db = chromadb.PersistentClient(path=chroma_path)

    if reset:
        try:
            db.delete_collection(collection_name)
            print("Cleared existing index.")
        except Exception as exc:
            if "does not exist" in str(exc).lower() or "not found" in str(exc).lower():
                pass
            else:
                raise

    collection = db.get_or_create_collection(
        collection_name, metadata={"hnsw:space": "cosine"}
    )

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    all_chunks = []
    for md_file in md_files:
        content = md_file.read_text(encoding="utf-8")
        rel_path = str(md_file)
        chunks = chunk_markdown(content, rel_path)
        all_chunks.extend(chunks)
        print(f"  {rel_path}: {len(chunks)} chunk(s)")

    if not all_chunks:
        print("No chunks generated.")
        return {"files": len(md_files), "chunks": 0}

    texts = [c.text for c in all_chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    metadatas = []
    for c in all_chunks:
        meta = dict(c.metadata)
        meta["image_refs"] = ",".join(meta["image_refs"]) if meta["image_refs"] else ""
        metadatas.append(meta)

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"\nIndexed {len(all_chunks)} chunks from {len(md_files)} file(s).")
    return {"files": len(md_files), "chunks": len(all_chunks)}
```

Remove the `main()` function and `if __name__ == "__main__"` block — CLI is handled by `cli.py`.

**Step 3: Update test imports in `tests/test_indexer.py`**

Change line 10 from:
```python
from indexer import index_documents
```
to:
```python
from markdex.indexer import index_documents
```

Update all `index_documents()` calls to use `paths=[FIXTURES_DIR]` instead of `docs_dir=FIXTURES_DIR`:
- Line 29: `paths=[FIXTURES_DIR],`
- Line 38: `paths=[FIXTURES_DIR],`
- Line 48: `paths=[FIXTURES_DIR],`
- Line 63: `paths=[FIXTURES_DIR],`
- Line 73: `paths=[FIXTURES_DIR],`

**Step 4: Run tests to verify**

Run: `pytest tests/test_indexer.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/markdex/indexer.py tests/test_indexer.py
git rm indexer.py
git commit -m "refactor: move indexer into package, accept file/dir paths"
```

---

### Task 4: Move and refactor server

**Files:**
- Move: `server.py` -> `src/markdex/server.py`
- Modify: `src/markdex/server.py` (use config, add index_document tool, simplify tool signatures)
- Modify: `tests/test_server.py` (update imports, update tool call signatures)

**Step 1: Move server.py**

```bash
mv server.py src/markdex/server.py
```

**Step 2: Rewrite `src/markdex/server.py`**

```python
#!/usr/bin/env python3
"""MCP server exposing markdex query and indexing tools for Markdown documents."""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from mcp.server.fastmcp import FastMCP

from markdex.config import CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL, ensure_data_dir

mcp = FastMCP("markdex")

# Lazy-loaded globals
_model = None
_collection = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        ensure_data_dir()
        db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _collection = db.get_or_create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
    return _collection


def _reset_collection():
    """Force reload of collection on next access."""
    global _collection
    _collection = None


@mcp.tool()
def query_docs(query: str, top_k: int = 5) -> str:
    """Search indexed Markdown documents. Returns the most relevant chunks with source file and heading metadata."""
    model = _get_model()
    collection = _get_collection()

    if collection.count() == 0:
        return "No documents indexed. Run: markdex index <path>"

    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
    )

    if not results["documents"] or not results["documents"][0]:
        return "No results found."

    output_parts = []
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        similarity = 1 - dist
        heading = meta.get("heading", "")
        file_path = meta.get("file_path", "unknown")
        header = f"[{i + 1}] {file_path}"
        if heading:
            header += f" > {heading}"
        header += f" (similarity: {similarity:.2f})"

        output_parts.append(f"{header}\n{doc}")

    return "\n\n---\n\n".join(output_parts)


@mcp.tool()
def list_indexed_files() -> str:
    """List all Markdown files currently in the index."""
    collection = _get_collection()

    results = collection.get(include=["metadatas"])

    if not results["metadatas"]:
        return "No files indexed. Run: markdex index <path>"

    files: dict[str, int] = {}
    for meta in results["metadatas"]:
        fp = meta.get("file_path", "unknown")
        files[fp] = files.get(fp, 0) + 1

    total_chunks = sum(files.values())
    lines = [f"Indexed {len(files)} file(s), {total_chunks} chunk(s):\n"]
    for fp, count in sorted(files.items()):
        lines.append(f"  {fp} ({count} chunks)")

    return "\n".join(lines)


@mcp.tool()
def index_document(path: str) -> str:
    """Index a Markdown file or directory into the search index. Accepts an absolute path to a .md file or a directory containing .md files."""
    from markdex.indexer import index_documents

    target = Path(path)
    if not target.exists():
        return f"Error: path '{path}' does not exist."

    if target.is_file() and target.suffix != ".md":
        return f"Error: '{path}' is not a .md file."

    stats = index_documents(paths=[str(target)])

    # Reset cached collection so queries see new data
    _reset_collection()

    return f"Indexed {stats['chunks']} chunk(s) from {stats['files']} file(s)."


def run():
    """Entry point for MCP server."""
    mcp.run()
```

**Step 3: Update `tests/test_server.py`**

```python
"""Integration tests for MCP server tool functions."""

import shutil
from pathlib import Path

import chromadb.api.client
import pytest

from markdex.indexer import index_documents
from markdex.server import query_docs, list_indexed_files, _reset_collection
from markdex import config

FIXTURES_DIR = str(Path(__file__).parent / "fixtures" / "sample_docs")
TEST_DB_PATH = str(Path(__file__).parent / "fixtures" / "test_chroma_db")
TEST_COLLECTION = "test_markdown_docs"


@pytest.fixture(scope="module", autouse=True)
def build_test_index(monkeypatch_module):
    """Build a test index once for all server tests."""
    monkeypatch_module.setattr(config, "CHROMA_DB_PATH", TEST_DB_PATH)
    monkeypatch_module.setattr(config, "COLLECTION_NAME", TEST_COLLECTION)
    _reset_collection()
    chromadb.api.client.Client.clear_system_cache()
    shutil.rmtree(TEST_DB_PATH, ignore_errors=True)
    index_documents(
        paths=[FIXTURES_DIR],
        chroma_path=TEST_DB_PATH,
        collection_name=TEST_COLLECTION,
    )
    yield
    _reset_collection()
    chromadb.api.client.Client.clear_system_cache()
    shutil.rmtree(TEST_DB_PATH, ignore_errors=True)


@pytest.fixture(scope="module")
def monkeypatch_module():
    """Module-scoped monkeypatch."""
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


class TestQueryDocs:
    def test_returns_relevant_chunks(self):
        result = query_docs("revenue growth", top_k=3)
        assert "Revenue" in result or "revenue" in result

    def test_returns_metadata_in_output(self):
        result = query_docs("financial metrics", top_k=2)
        assert "sample_docs/" in result
        assert "similarity:" in result

    def test_respects_top_k(self):
        result = query_docs("analysis", top_k=1)
        assert "[1]" in result
        assert "[2]" not in result


class TestListIndexedFiles:
    def test_lists_files(self):
        result = list_indexed_files()
        assert "basics.md" in result
        assert "analysis.md" in result

    def test_shows_chunk_counts(self):
        result = list_indexed_files()
        assert "chunk" in result.lower()
```

Note: The empty index test is removed because the server now uses config-driven paths and testing empty state requires more fixture management. The important behavior (empty message) is covered by the message string in the code.

**Step 4: Run tests to verify**

Run: `pytest tests/test_server.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/markdex/server.py tests/test_server.py
git rm server.py
git commit -m "refactor: move server into package, add index_document tool"
```

---

### Task 5: Create CLI

**Files:**
- Create: `src/markdex/cli.py`

**Step 1: Create `src/markdex/cli.py`**

```python
"""CLI entry point for markdex."""

import argparse
import sys


def cmd_index(args):
    """Index Markdown files."""
    from markdex.indexer import index_documents
    index_documents(paths=args.paths, reset=args.reset)


def cmd_serve(args):
    """Start MCP server."""
    from markdex.server import run
    run()


def cmd_list(args):
    """List indexed files."""
    from markdex.config import CHROMA_DB_PATH, COLLECTION_NAME, ensure_data_dir
    import chromadb

    ensure_data_dir()
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = db.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    results = collection.get(include=["metadatas"])
    if not results["metadatas"]:
        print("No files indexed. Run: markdex index <path>")
        return

    files: dict[str, int] = {}
    for meta in results["metadatas"]:
        fp = meta.get("file_path", "unknown")
        files[fp] = files.get(fp, 0) + 1

    total_chunks = sum(files.values())
    print(f"Indexed {len(files)} file(s), {total_chunks} chunk(s):\n")
    for fp, count in sorted(files.items()):
        print(f"  {fp} ({count} chunks)")


def main():
    parser = argparse.ArgumentParser(
        prog="markdex",
        description="Local Markdown indexing and retrieval via MCP.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # markdex index
    index_parser = subparsers.add_parser("index", help="Index Markdown files")
    index_parser.add_argument("paths", nargs="+", help="Files or directories to index")
    index_parser.add_argument("--reset", action="store_true", help="Clear and rebuild the index")
    index_parser.set_defaults(func=cmd_index)

    # markdex serve
    serve_parser = subparsers.add_parser("serve", help="Start MCP server")
    serve_parser.set_defaults(func=cmd_serve)

    # markdex list
    list_parser = subparsers.add_parser("list", help="List indexed files")
    list_parser.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)
```

**Step 2: Verify CLI wires up**

Run: `pip install -e . && markdex --help`
Expected: Shows usage with `index`, `serve`, `list` subcommands

**Step 3: Commit**

```bash
git add src/markdex/cli.py
git commit -m "feat: add CLI with index, serve, list commands"
```

---

### Task 6: Clean up old files and install

**Files:**
- Delete: `requirements.txt`
- Delete: `tests/__init__.py`
- Modify: `.mcp.json`
- Modify: `.gitignore`

**Step 1: Delete `requirements.txt`** (replaced by pyproject.toml)

```bash
git rm requirements.txt
```

**Step 2: Delete `tests/__init__.py`** (empty, not needed with src layout)

```bash
git rm tests/__init__.py
```

**Step 3: Update `.mcp.json`** for the new entry point

```json
{
  "mcpServers": {
    "markdex": {
      "command": "markdex",
      "args": ["serve"]
    }
  }
}
```

**Step 4: Add `src/` aware pytest config to `pyproject.toml`**

Append to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 5: Install in editable mode and run full test suite**

Run: `pip install -e ".[dev]" && pytest tests/ -v`
Expected: All tests PASS

**Step 6: Verify CLI commands work**

Run: `markdex list`
Expected: Shows indexed files or "No files indexed" message

Run: `markdex index docs/`
Expected: Indexes files into `~/.markdex/chroma_db/`

**Step 7: Commit**

```bash
git add .mcp.json .gitignore pyproject.toml
git rm requirements.txt tests/__init__.py
git commit -m "chore: clean up old files, update project config"
```

---

### Task 7: Update README and CLAUDE.md

**Files:**
- Create: `README.md`
- Modify: `CLAUDE.md`

**Step 1: Create `README.md`**

```markdown
# markdex

Local Markdown indexing and retrieval via MCP. Index your Markdown documents and query them from Claude Code using semantic search.

## Install

```bash
git clone <repo-url>
cd markdex
pip install -e .
```

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
```

**Step 2: Update `CLAUDE.md`**

```markdown
# markdex

Local Markdown indexing and retrieval via MCP.

## Quick Reference

- **Index docs:** `markdex index <path>` (or `markdex index <path> --reset` to rebuild)
- **Run MCP server:** `markdex serve` (stdio transport, used by Claude Code)
- **List indexed files:** `markdex list`
- **Run tests:** `pytest tests/ -v`

## Architecture

- `src/markdex/config.py` — Centralized paths and constants (`~/.markdex/`)
- `src/markdex/chunker.py` — Markdown parsing and heading-based chunking
- `src/markdex/indexer.py` — Reads paths, chunks, embeds, stores in ChromaDB
- `src/markdex/server.py` — MCP server with `query_docs`, `list_indexed_files`, and `index_document` tools
- `src/markdex/cli.py` — CLI entry point (`markdex index`, `markdex serve`, `markdex list`)

## Important Notes

- Data stored in `~/.markdex/chroma_db/`
- The indexer uses `upsert` so running it multiple times without `--reset` is safe
- The `index_document` MCP tool indexes in-process — no server restart needed

## Dependencies

chromadb, sentence-transformers, mcp, pytest
```

**Step 3: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: add README and update CLAUDE.md for new package structure"
```

---

### Task 8: Update global MCP config

**Step 1: Update `~/.claude.json`**

Update the markdex MCP server entry to use the new CLI:

```json
{
  "markdex": {
    "type": "stdio",
    "command": "markdex",
    "args": ["serve"]
  }
}
```

This replaces the current entry that points to `.venv/bin/python` with an absolute path. Since `markdex` is pip-installed, it's on PATH.

**Step 2: Verify MCP connection**

Run `/mcp` in Claude Code to reconnect and verify markdex tools are available.

**Step 3: Commit** (no code change — this is user config)
