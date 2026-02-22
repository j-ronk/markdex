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
