# markdex Design: Local Markdown + Image RAG for Claude Code via MCP

**Date:** 2026-02-20
**Status:** Approved

## Goal

Build a fully local RAG pipeline that indexes Markdown files (with optional images later) and exposes them to Claude Code via an MCP tool so queries return relevant document chunks as context.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Retrieval mode | Chunks only, no LLM synthesis | Claude Code IS the LLM. Returning raw chunks lets Claude reason over the actual text. |
| Framework | Direct (no LlamaIndex/LangChain) | ~150 lines of our own code vs 50+ transitive deps. Full control, easy to debug. |
| Embedding model | `all-mpnet-base-v2` | Highest quality general-purpose sentence-transformers model. Speed irrelevant at <50 files. |
| Vector store | ChromaDB (persistent) | Simple, file-based, no server process needed. Good Python API. |
| Architecture | Single-file MCP server + separate indexer | Two clear entry points. Right-sized for <50 files. |
| MCP transport | stdio | Standard transport for local MCP servers with Claude Code. |

## Project Structure

```
rag-pipeline/
├── docs/                  # Markdown files to index
│   └── images/            # Optional image folder
├── chroma_db/             # Persisted ChromaDB data (git-ignored)
├── server.py              # MCP server with query tools
├── indexer.py             # Build/rebuild the index
├── requirements.txt       # chromadb, sentence-transformers, mcp
├── .gitignore
└── CLAUDE.md
```

## Dependencies

```
chromadb>=0.5.0
sentence-transformers>=3.0.0
mcp>=1.0.0
```

## Markdown Chunking Strategy

Split on Markdown headings (`#`, `##`, `###`, etc.). Each heading starts a new chunk. Content under a heading (paragraphs, bullets, tables) stays grouped.

**Chunk metadata:**
- `file_path` — source file
- `heading` — heading text
- `heading_level` — 1, 2, 3, etc.
- `image_refs` — image paths referenced in the chunk (for future use)
- `file_hash` — MD5 of source file (for future incremental indexing)

**Size handling:**
- Sections >1000 chars: split further at paragraph boundaries
- Sections <50 chars: merge with next section
- Tables and code blocks: kept intact, never split mid-block

## MCP Server (`server.py`)

Uses official `mcp` Python SDK with `FastMCP`. Runs over stdio.

### Tools

**`query_docs(query: str, top_k: int = 5) -> str`**
Search indexed Markdown documents. Returns the most relevant chunks with source file and heading metadata.

Response format:
```
[1] docs/q4-analysis.md > Q4 Revenue (similarity: 0.87)
Revenue grew 12% YoY driven by strong performance in APAC...

[2] docs/annual-report.md > Financial Summary (similarity: 0.82)
Total revenue reached $2.1B, a record high...
```

**`list_indexed_files() -> str`**
List all Markdown files currently in the index with chunk counts.

### Claude Code Configuration

```json
{
  "mcpServers": {
    "markdown-rag": {
      "command": "/path/to/rag-pipeline/.venv/bin/python",
      "args": ["/path/to/rag-pipeline/server.py"]
    }
  }
}
```

## Indexer (`indexer.py`)

Standalone script to build or rebuild the index.

```
python indexer.py                    # Index all .md files in ./docs/
python indexer.py --docs-dir /path   # Custom docs directory
python indexer.py --reset            # Clear and rebuild from scratch
```

**Process:**
1. Scan `docs/` for all `.md` files (recursive)
2. Read each file, parse into heading-based chunks
3. Attach metadata (file_path, heading, heading_level, image_refs)
4. Embed all chunks using `all-mpnet-base-v2`
5. Store in ChromaDB at `./chroma_db/`
6. Print summary: X files, Y chunks indexed

**Rebuild strategy:** Full clear-and-rebuild for v1. Incremental indexing (hash-based dedup) deferred until scale requires it.

## Data Flow

```
User query → Claude Code → MCP tool (query_docs)
                                ↓
                         Embed query with all-mpnet-base-v2
                                ↓
                         ChromaDB similarity search (top-k)
                                ↓
                         Format results with metadata
                                ↓
                         Return chunk text → Claude Code
                                ↓
                         Claude reasons over chunks → answer
```

## Future Enhancements (Not in v1)

- Incremental indexing with file hashing
- Image retrieval via metadata filtering
- Custom chunking for tables and code blocks
- Semantic filters (by tag, section, file)
- Re-indexing MCP tool (so Claude Code can trigger rebuilds)
