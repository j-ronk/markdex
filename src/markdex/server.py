#!/usr/bin/env python3
"""MCP server exposing markdex query and indexing tools for Markdown documents."""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from mcp.server.fastmcp import FastMCP

from markdex import config
from markdex.config import ensure_data_dir

mcp = FastMCP("markdex")

# Lazy-loaded globals
_model = None
_collection = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        ensure_data_dir()
        db = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        _collection = db.get_or_create_collection(
            config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
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
