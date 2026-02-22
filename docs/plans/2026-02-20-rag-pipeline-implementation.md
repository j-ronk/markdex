# markdex Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a local RAG pipeline that indexes Markdown files into ChromaDB and exposes retrieval to Claude Code via MCP tools.

**Architecture:** Two entry points — `indexer.py` builds the vector index from Markdown files, `server.py` serves queries via MCP stdio. A separate `chunker.py` module handles Markdown parsing. No LLM synthesis; raw chunks are returned for Claude Code to reason over.

**Tech Stack:** Python 3.13, chromadb, sentence-transformers (all-mpnet-base-v2), mcp (FastMCP), pytest

---

### Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `CLAUDE.md`
- Create: `docs/.gitkeep`

**Step 1: Create requirements.txt**

```
chromadb>=0.5.0
sentence-transformers>=3.0.0
mcp>=1.0.0
pytest>=8.0.0
```

**Step 2: Create .gitignore**

```
# Python
__pycache__/
*.pyc
.venv/

# ChromaDB
chroma_db/

# IDE
.idea/

# OS
.DS_Store
```

**Step 3: Create CLAUDE.md**

```markdown
# Markdown RAG Pipeline

Local RAG pipeline for indexing Markdown files and exposing retrieval to Claude Code via MCP.

## Quick Reference

- **Index docs:** `python indexer.py` (or `python indexer.py --reset` to rebuild)
- **Run MCP server:** `python server.py` (stdio transport, used by Claude Code)
- **Run tests:** `pytest tests/ -v`

## Architecture

- `chunker.py` — Markdown parsing and heading-based chunking
- `indexer.py` — Reads docs/, chunks, embeds, stores in ChromaDB
- `server.py` — MCP server with `query_docs` and `list_indexed_files` tools
- `chroma_db/` — Persistent vector store (git-ignored)
- `docs/` — Markdown files to index

## Key Constants (duplicated in indexer.py and server.py)

- `CHROMA_DB_PATH = "./chroma_db"`
- `COLLECTION_NAME = "markdown_docs"`
- `EMBEDDING_MODEL = "all-mpnet-base-v2"`

## Dependencies

chromadb, sentence-transformers, mcp, pytest
```

**Step 4: Create docs/.gitkeep**

Empty file so the docs/ directory is tracked by git.

**Step 5: Install dependencies**

Run: `.venv/bin/pip install -r requirements.txt`

Expected: All packages install successfully. sentence-transformers pulls in PyTorch.

**Step 6: Commit**

```bash
git add requirements.txt .gitignore CLAUDE.md docs/.gitkeep
git commit -m "feat: project scaffolding with dependencies"
```

---

### Task 2: Markdown Chunker — Tests

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_chunker.py`

**Step 1: Create tests/__init__.py**

Empty file.

**Step 2: Write failing tests for chunker**

```python
"""Tests for Markdown chunking logic."""

from chunker import chunk_markdown, Chunk


class TestChunkMarkdownBasic:
    """Test basic heading-based splitting."""

    def test_single_heading_with_content(self):
        content = "# Introduction\n\nThis is the intro paragraph."
        chunks = chunk_markdown(content, "test.md")
        assert len(chunks) == 1
        assert chunks[0].metadata["heading"] == "Introduction"
        assert chunks[0].metadata["heading_level"] == 1
        assert chunks[0].metadata["file_path"] == "test.md"
        assert "intro paragraph" in chunks[0].text

    def test_multiple_headings(self):
        content = "# Section A\n\nContent A.\n\n## Section B\n\nContent B."
        chunks = chunk_markdown(content, "test.md")
        assert len(chunks) == 2
        assert chunks[0].metadata["heading"] == "Section A"
        assert chunks[0].metadata["heading_level"] == 1
        assert chunks[1].metadata["heading"] == "Section B"
        assert chunks[1].metadata["heading_level"] == 2

    def test_no_headings(self):
        content = "Just some text without any headings.\n\nAnother paragraph."
        chunks = chunk_markdown(content, "test.md")
        assert len(chunks) == 1
        assert chunks[0].metadata["heading"] == ""
        assert chunks[0].metadata["heading_level"] == 0

    def test_content_before_first_heading(self):
        content = "Preamble text here.\n\n# First Heading\n\nHeading content."
        chunks = chunk_markdown(content, "test.md")
        assert len(chunks) == 2
        assert chunks[0].metadata["heading"] == ""
        assert "Preamble" in chunks[0].text
        assert chunks[1].metadata["heading"] == "First Heading"

    def test_empty_content(self):
        chunks = chunk_markdown("", "test.md")
        assert len(chunks) == 0


class TestChunkMetadata:
    """Test metadata extraction."""

    def test_file_path_preserved(self):
        content = "# Title\n\nBody text."
        chunks = chunk_markdown(content, "docs/finance/q4.md")
        assert chunks[0].metadata["file_path"] == "docs/finance/q4.md"

    def test_file_hash_computed(self):
        content = "# Title\n\nBody text."
        chunks = chunk_markdown(content, "test.md")
        assert len(chunks[0].metadata["file_hash"]) == 32  # MD5 hex length

    def test_file_hash_changes_with_content(self):
        chunks_a = chunk_markdown("# A\n\nContent A.", "test.md")
        chunks_b = chunk_markdown("# B\n\nContent B.", "test.md")
        assert chunks_a[0].metadata["file_hash"] != chunks_b[0].metadata["file_hash"]

    def test_image_refs_extracted(self):
        content = "# Diagram\n\nSee below:\n\n![Chart](./images/chart.png)"
        chunks = chunk_markdown(content, "test.md")
        assert chunks[0].metadata["image_refs"] == ["./images/chart.png"]

    def test_multiple_image_refs(self):
        content = "# Images\n\n![A](a.png)\n\nText\n\n![B](b.png)"
        chunks = chunk_markdown(content, "test.md")
        assert chunks[0].metadata["image_refs"] == ["a.png", "b.png"]

    def test_no_image_refs(self):
        content = "# Plain\n\nNo images here."
        chunks = chunk_markdown(content, "test.md")
        assert chunks[0].metadata["image_refs"] == []


class TestChunkSizeHandling:
    """Test large section splitting and small section merging."""

    def test_large_section_split_at_paragraphs(self):
        # Create content > 1000 chars under one heading
        long_para_1 = "A" * 600
        long_para_2 = "B" * 600
        content = f"# Big Section\n\n{long_para_1}\n\n{long_para_2}"
        chunks = chunk_markdown(content, "test.md")
        assert len(chunks) >= 2
        for c in chunks:
            assert c.metadata["heading"] == "Big Section"

    def test_small_section_merged_with_next(self):
        content = "# Tiny\n\nHi\n\n# Normal\n\nThis is a normal length section with enough content."
        chunks = chunk_markdown(content, "test.md")
        # "Tiny" section ("# Tiny\n\nHi") is < 50 chars, should merge with next
        assert len(chunks) == 1
        assert "Hi" in chunks[0].text
        assert "normal length" in chunks[0].text

    def test_table_kept_intact(self):
        content = "# Data\n\n| Col A | Col B |\n|-------|-------|\n| 1     | 2     |\n| 3     | 4     |"
        chunks = chunk_markdown(content, "test.md")
        assert len(chunks) == 1
        assert "| Col A | Col B |" in chunks[0].text
        assert "| 3     | 4     |" in chunks[0].text
```

**Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_chunker.py -v`

Expected: `ModuleNotFoundError: No module named 'chunker'` — all tests fail because chunker.py doesn't exist yet.

**Step 4: Commit**

```bash
git add tests/
git commit -m "test: add chunker unit tests (red phase)"
```

---

### Task 3: Markdown Chunker — Implementation

**Files:**
- Create: `chunker.py`

**Step 1: Implement the chunker**

```python
"""Markdown chunking: split files into semantically meaningful chunks by headings."""

import hashlib
import re
from dataclasses import dataclass, field

MAX_CHUNK_CHARS = 1000
MIN_CHUNK_CHARS = 50
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
IMAGE_PATTERN = re.compile(r"!\[.*?\]\((.+?)\)")


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


def chunk_markdown(content: str, file_path: str) -> list[Chunk]:
    """Split markdown content into chunks based on headings.

    Each heading starts a new chunk. Sections >1000 chars are split at paragraph
    boundaries. Sections <50 chars are merged with the following section.
    """
    content = content.strip()
    if not content:
        return []

    file_hash = hashlib.md5(content.encode()).hexdigest()
    matches = list(HEADING_PATTERN.finditer(content))

    if not matches:
        return [_make_chunk(content, file_path, "", 0, file_hash)]

    raw_chunks: list[Chunk] = []

    # Content before first heading
    if matches[0].start() > 0:
        pre = content[: matches[0].start()].strip()
        if pre:
            raw_chunks.append(_make_chunk(pre, file_path, "", 0, file_hash))

    # Each heading section
    for i, match in enumerate(matches):
        level = len(match.group(1))
        heading = match.group(2).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        section = content[start:end].strip()

        if len(section) > MAX_CHUNK_CHARS:
            raw_chunks.extend(_split_large(section, file_path, heading, level, file_hash))
        else:
            raw_chunks.append(_make_chunk(section, file_path, heading, level, file_hash))

    return _merge_small(raw_chunks)


def _make_chunk(text: str, file_path: str, heading: str, level: int, file_hash: str) -> Chunk:
    return Chunk(
        text=text,
        metadata={
            "file_path": file_path,
            "heading": heading,
            "heading_level": level,
            "image_refs": IMAGE_PATTERN.findall(text),
            "file_hash": file_hash,
        },
    )


def _split_large(text: str, file_path: str, heading: str, level: int, file_hash: str) -> list[Chunk]:
    """Split a large section at paragraph boundaries (double newline)."""
    paragraphs = re.split(r"\n\n+", text)
    chunks: list[Chunk] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) > MAX_CHUNK_CHARS and current:
            chunks.append(_make_chunk(current.strip(), file_path, heading, level, file_hash))
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para

    if current.strip():
        chunks.append(_make_chunk(current.strip(), file_path, heading, level, file_hash))

    return chunks


def _merge_small(chunks: list[Chunk]) -> list[Chunk]:
    """Merge chunks smaller than MIN_CHUNK_CHARS with the next chunk."""
    if len(chunks) <= 1:
        return chunks

    merged: list[Chunk] = []
    i = 0
    while i < len(chunks):
        if len(chunks[i].text) < MIN_CHUNK_CHARS and i + 1 < len(chunks):
            combined_text = chunks[i].text + "\n\n" + chunks[i + 1].text
            merged.append(Chunk(text=combined_text, metadata=chunks[i + 1].metadata))
            i += 2
        else:
            merged.append(chunks[i])
            i += 1

    return merged
```

**Step 2: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_chunker.py -v`

Expected: All tests PASS.

**Step 3: Commit**

```bash
git add chunker.py
git commit -m "feat: markdown chunker with heading-based splitting"
```

---

### Task 4: Indexer — Test

**Files:**
- Create: `tests/test_indexer.py`
- Create: `tests/fixtures/sample_docs/basics.md`
- Create: `tests/fixtures/sample_docs/analysis.md`

**Step 1: Create test fixture files**

`tests/fixtures/sample_docs/basics.md`:
```markdown
# Getting Started

This document covers the basics of our financial analysis framework.

## Key Metrics

We track revenue, EBITDA, and free cash flow as our primary metrics.

## Data Sources

All data comes from SEC filings and Bloomberg terminals.
```

`tests/fixtures/sample_docs/analysis.md`:
```markdown
# Q4 Revenue Analysis

Revenue grew 12% YoY driven by strong performance in APAC.

## Regional Breakdown

| Region | Revenue | Growth |
|--------|---------|--------|
| NA     | $500M   | 8%     |
| APAC   | $300M   | 22%    |
| EMEA   | $200M   | 5%     |

## Outlook

We expect continued momentum in Q1.

![Revenue Chart](./images/revenue.png)
```

**Step 2: Write integration test for indexer**

```python
"""Integration tests for the indexer (requires chromadb + sentence-transformers)."""

import shutil
from pathlib import Path

import chromadb
import pytest

from indexer import index_documents

FIXTURES_DIR = str(Path(__file__).parent / "fixtures" / "sample_docs")
TEST_DB_PATH = str(Path(__file__).parent / "fixtures" / "test_chroma_db")
TEST_COLLECTION = "test_markdown_docs"


@pytest.fixture(autouse=True)
def clean_test_db():
    """Remove test ChromaDB before and after each test."""
    shutil.rmtree(TEST_DB_PATH, ignore_errors=True)
    yield
    shutil.rmtree(TEST_DB_PATH, ignore_errors=True)


class TestIndexer:
    def test_indexes_all_files(self):
        stats = index_documents(
            docs_dir=FIXTURES_DIR,
            chroma_path=TEST_DB_PATH,
            collection_name=TEST_COLLECTION,
        )
        assert stats["files"] == 2
        assert stats["chunks"] > 0

    def test_chunks_stored_in_chromadb(self):
        index_documents(
            docs_dir=FIXTURES_DIR,
            chroma_path=TEST_DB_PATH,
            collection_name=TEST_COLLECTION,
        )
        db = chromadb.PersistentClient(path=TEST_DB_PATH)
        collection = db.get_or_create_collection(TEST_COLLECTION)
        assert collection.count() > 0

    def test_metadata_stored(self):
        index_documents(
            docs_dir=FIXTURES_DIR,
            chroma_path=TEST_DB_PATH,
            collection_name=TEST_COLLECTION,
        )
        db = chromadb.PersistentClient(path=TEST_DB_PATH)
        collection = db.get_or_create_collection(TEST_COLLECTION)
        results = collection.get(include=["metadatas"])
        meta = results["metadatas"][0]
        assert "file_path" in meta
        assert "heading" in meta
        assert "heading_level" in meta

    def test_reset_clears_and_rebuilds(self):
        # Index once
        index_documents(
            docs_dir=FIXTURES_DIR,
            chroma_path=TEST_DB_PATH,
            collection_name=TEST_COLLECTION,
        )
        db = chromadb.PersistentClient(path=TEST_DB_PATH)
        collection = db.get_or_create_collection(TEST_COLLECTION)
        first_count = collection.count()

        # Index again with reset
        stats = index_documents(
            docs_dir=FIXTURES_DIR,
            chroma_path=TEST_DB_PATH,
            collection_name=TEST_COLLECTION,
            reset=True,
        )
        db2 = chromadb.PersistentClient(path=TEST_DB_PATH)
        collection2 = db2.get_or_create_collection(TEST_COLLECTION)
        assert collection2.count() == first_count  # same files, same count
        assert stats["files"] == 2
```

**Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_indexer.py -v`

Expected: `ModuleNotFoundError: No module named 'indexer'`

**Step 4: Commit**

```bash
git add tests/
git commit -m "test: add indexer integration tests (red phase)"
```

---

### Task 5: Indexer — Implementation

**Files:**
- Create: `indexer.py`

**Step 1: Implement the indexer**

```python
#!/usr/bin/env python3
"""Index Markdown files into ChromaDB for RAG retrieval."""

import argparse
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from chunker import chunk_markdown

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "markdown_docs"
EMBEDDING_MODEL = "all-mpnet-base-v2"
DOCS_DIR = "./docs"


def index_documents(
    docs_dir: str = DOCS_DIR,
    chroma_path: str = CHROMA_DB_PATH,
    collection_name: str = COLLECTION_NAME,
    reset: bool = False,
) -> dict:
    """Index all Markdown files in docs_dir into ChromaDB.

    Returns dict with keys: files, chunks.
    """
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        print(f"Error: docs directory '{docs_dir}' does not exist.")
        sys.exit(1)

    md_files = sorted(docs_path.rglob("*.md"))
    if not md_files:
        print(f"No .md files found in '{docs_dir}'.")
        sys.exit(1)

    print(f"Found {len(md_files)} Markdown file(s)")

    db = chromadb.PersistentClient(path=chroma_path)

    if reset:
        try:
            db.delete_collection(collection_name)
            print("Cleared existing index.")
        except ValueError:
            pass

    collection = db.get_or_create_collection(
        collection_name, metadata={"hnsw:space": "cosine"}
    )

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    all_chunks = []
    for md_file in md_files:
        content = md_file.read_text(encoding="utf-8")
        rel_path = str(md_file.relative_to(docs_path.parent))
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
        # ChromaDB metadata values must be str, int, float, or bool
        meta["image_refs"] = ",".join(meta["image_refs"]) if meta["image_refs"] else ""
        metadatas.append(meta)

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"\nIndexed {len(all_chunks)} chunks from {len(md_files)} file(s).")
    return {"files": len(md_files), "chunks": len(all_chunks)}


def main():
    parser = argparse.ArgumentParser(description="Index Markdown files for RAG retrieval.")
    parser.add_argument("--docs-dir", default=DOCS_DIR, help="Directory containing Markdown files")
    parser.add_argument("--reset", action="store_true", help="Clear and rebuild the index")
    args = parser.parse_args()

    index_documents(docs_dir=args.docs_dir, reset=args.reset)


if __name__ == "__main__":
    main()
```

**Step 2: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_indexer.py -v`

Expected: All tests PASS. First run will download the embedding model (~420MB).

**Step 3: Commit**

```bash
git add indexer.py
git commit -m "feat: indexer script for Markdown to ChromaDB pipeline"
```

---

### Task 6: MCP Server — Test

**Files:**
- Create: `tests/test_server.py`

**Step 1: Write tests for server tool functions**

We test the tool functions directly (no MCP transport needed).

```python
"""Integration tests for MCP server tool functions."""

import shutil
from pathlib import Path

import pytest

from indexer import index_documents
from server import query_docs, list_indexed_files

FIXTURES_DIR = str(Path(__file__).parent / "fixtures" / "sample_docs")
TEST_DB_PATH = str(Path(__file__).parent / "fixtures" / "test_chroma_db")
TEST_COLLECTION = "test_markdown_docs"


@pytest.fixture(scope="module", autouse=True)
def build_test_index():
    """Build a test index once for all server tests."""
    shutil.rmtree(TEST_DB_PATH, ignore_errors=True)
    index_documents(
        docs_dir=FIXTURES_DIR,
        chroma_path=TEST_DB_PATH,
        collection_name=TEST_COLLECTION,
    )
    yield
    shutil.rmtree(TEST_DB_PATH, ignore_errors=True)


class TestQueryDocs:
    def test_returns_relevant_chunks(self):
        result = query_docs(
            "revenue growth",
            top_k=3,
            chroma_path=TEST_DB_PATH,
            collection_name=TEST_COLLECTION,
        )
        assert "Revenue" in result or "revenue" in result

    def test_returns_metadata_in_output(self):
        result = query_docs(
            "financial metrics",
            top_k=2,
            chroma_path=TEST_DB_PATH,
            collection_name=TEST_COLLECTION,
        )
        # Output should contain file path and similarity info
        assert "sample_docs/" in result
        assert "similarity:" in result

    def test_respects_top_k(self):
        result = query_docs(
            "analysis",
            top_k=1,
            chroma_path=TEST_DB_PATH,
            collection_name=TEST_COLLECTION,
        )
        # Should only have one result marker [1], not [2]
        assert "[1]" in result
        assert "[2]" not in result

    def test_empty_index_returns_message(self):
        empty_path = str(Path(__file__).parent / "fixtures" / "empty_chroma_db")
        shutil.rmtree(empty_path, ignore_errors=True)
        try:
            result = query_docs(
                "anything",
                chroma_path=empty_path,
                collection_name="nonexistent",
            )
            assert "No results" in result or "no" in result.lower()
        finally:
            shutil.rmtree(empty_path, ignore_errors=True)


class TestListIndexedFiles:
    def test_lists_files(self):
        result = list_indexed_files(
            chroma_path=TEST_DB_PATH,
            collection_name=TEST_COLLECTION,
        )
        assert "basics.md" in result
        assert "analysis.md" in result

    def test_shows_chunk_counts(self):
        result = list_indexed_files(
            chroma_path=TEST_DB_PATH,
            collection_name=TEST_COLLECTION,
        )
        assert "chunk" in result.lower()
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_server.py -v`

Expected: `ModuleNotFoundError: No module named 'server'`

**Step 3: Commit**

```bash
git add tests/test_server.py
git commit -m "test: add MCP server tool function tests (red phase)"
```

---

### Task 7: MCP Server — Implementation

**Files:**
- Create: `server.py`

**Step 1: Implement the MCP server**

```python
#!/usr/bin/env python3
"""MCP server exposing RAG query tools for Markdown documents."""

import chromadb
from sentence_transformers import SentenceTransformer
from mcp.server.fastmcp import FastMCP

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "markdown_docs"
EMBEDDING_MODEL = "all-mpnet-base-v2"

mcp = FastMCP("markdown-rag")

# Lazy-loaded globals
_model = None
_collections: dict = {}


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection(chroma_path: str = CHROMA_DB_PATH, collection_name: str = COLLECTION_NAME):
    key = f"{chroma_path}:{collection_name}"
    if key not in _collections:
        db = chromadb.PersistentClient(path=chroma_path)
        _collections[key] = db.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"}
        )
    return _collections[key]


@mcp.tool()
def query_docs(
    query: str,
    top_k: int = 5,
    chroma_path: str = CHROMA_DB_PATH,
    collection_name: str = COLLECTION_NAME,
) -> str:
    """Search indexed Markdown documents. Returns the most relevant chunks with source file and heading metadata."""
    model = _get_model()
    collection = _get_collection(chroma_path, collection_name)

    if collection.count() == 0:
        return "No results found. Is the index built? Run: python indexer.py"

    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
    )

    if not results["documents"] or not results["documents"][0]:
        return "No results found. Is the index built? Run: python indexer.py"

    output_parts = []
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        similarity = 1 - dist  # cosine distance: 0=identical, 2=opposite
        heading = meta.get("heading", "")
        file_path = meta.get("file_path", "unknown")
        header = f"[{i + 1}] {file_path}"
        if heading:
            header += f" > {heading}"
        header += f" (similarity: {similarity:.2f})"

        output_parts.append(f"{header}\n{doc}")

    return "\n\n---\n\n".join(output_parts)


@mcp.tool()
def list_indexed_files(
    chroma_path: str = CHROMA_DB_PATH,
    collection_name: str = COLLECTION_NAME,
) -> str:
    """List all Markdown files currently in the index."""
    collection = _get_collection(chroma_path, collection_name)

    results = collection.get(include=["metadatas"])

    if not results["metadatas"]:
        return "No files indexed. Run: python indexer.py"

    files: dict[str, int] = {}
    for meta in results["metadatas"]:
        fp = meta.get("file_path", "unknown")
        files[fp] = files.get(fp, 0) + 1

    total_chunks = sum(files.values())
    lines = [f"Indexed {len(files)} file(s), {total_chunks} chunk(s):\n"]
    for fp, count in sorted(files.items()):
        lines.append(f"  {fp} ({count} chunks)")

    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
```

**Step 2: Run server tests to verify they pass**

Run: `.venv/bin/pytest tests/test_server.py -v`

Expected: All tests PASS.

**Step 3: Run ALL tests to verify nothing is broken**

Run: `.venv/bin/pytest tests/ -v`

Expected: All tests in test_chunker.py, test_indexer.py, and test_server.py PASS.

**Step 4: Commit**

```bash
git add server.py
git commit -m "feat: MCP server with query_docs and list_indexed_files tools"
```

---

### Task 8: End-to-End Verification

**Files:**
- Modify: none (verification only)

**Step 1: Index the sample docs manually**

Run: `.venv/bin/python indexer.py --docs-dir tests/fixtures/sample_docs --reset`

Expected output:
```
Found 2 Markdown file(s)
Loading embedding model: all-mpnet-base-v2
  sample_docs/analysis.md: X chunk(s)
  sample_docs/basics.md: Y chunk(s)
Embedding N chunks...
Indexed N chunks from 2 file(s).
```

**Step 2: Test MCP server starts without error**

Run: `echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{},"clientInfo":{"name":"test"},"protocolVersion":"2024-11-05"}}' | .venv/bin/python server.py`

Expected: Server responds with JSON (capabilities listing). It should not crash.

**Step 3: Run full test suite one final time**

Run: `.venv/bin/pytest tests/ -v`

Expected: All tests PASS.

**Step 4: Commit any remaining changes**

```bash
git add -A
git commit -m "chore: end-to-end verification complete"
```

---

### Task 9: Claude Code MCP Configuration

**Files:**
- Create or modify: `.mcp.json` (project-level MCP config)

**Step 1: Create project-level MCP config**

Create `.mcp.json` in the project root:

```json
{
  "mcpServers": {
    "markdown-rag": {
      "command": ".venv/bin/python",
      "args": ["server.py"]
    }
  }
}
```

Note: This uses relative paths which work when Claude Code is invoked from the project root. For global access, use absolute paths in `~/.claude.json` instead.

**Step 2: Verify Claude Code sees the tool**

Open Claude Code in the project directory and check that the `markdown-rag` MCP server is listed. You should be able to call `query_docs` and `list_indexed_files`.

**Step 3: Commit**

```bash
git add .mcp.json
git commit -m "feat: add MCP config for Claude Code integration"
```

---

## Summary

| Task | What | Files |
|------|------|-------|
| 1 | Project scaffolding | requirements.txt, .gitignore, CLAUDE.md |
| 2 | Chunker tests (red) | tests/test_chunker.py |
| 3 | Chunker implementation (green) | chunker.py |
| 4 | Indexer tests (red) | tests/test_indexer.py, fixtures |
| 5 | Indexer implementation (green) | indexer.py |
| 6 | Server tests (red) | tests/test_server.py |
| 7 | Server implementation (green) | server.py |
| 8 | End-to-end verification | — |
| 9 | Claude Code MCP config | .mcp.json |

Total: 9 tasks, ~6 files of production code, ~3 test files, TDD throughout.
