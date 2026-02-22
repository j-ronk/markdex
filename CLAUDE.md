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
