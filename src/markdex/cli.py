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
    from markdex.indexer import list_indexed

    entries = list_indexed()
    if not entries:
        print("No files indexed. Run: markdex index <path>")
        return

    total_chunks = sum(count for _, count in entries)
    print(f"Indexed {len(entries)} file(s), {total_chunks} chunk(s):\n")
    for i, (fp, count) in enumerate(entries, 1):
        print(f"  [{i}] {fp} ({count} chunks)")


def cmd_remove(args):
    """Remove a file from the index by its number in `markdex list`."""
    from markdex.indexer import list_indexed, remove_from_index

    entries = list_indexed()
    if not entries:
        print("No files indexed.")
        sys.exit(1)

    num = args.number
    if num < 1 or num > len(entries):
        print(f"Invalid number {num}. Run `markdex list` to see valid entries (1-{len(entries)}).")
        sys.exit(1)

    file_path, _ = entries[num - 1]
    removed = remove_from_index(file_path)
    print(f"Removed {removed} chunk(s) for {file_path}")


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

    # markdex remove
    remove_parser = subparsers.add_parser("remove", help="Remove a file from the index")
    remove_parser.add_argument("number", type=int, help="File number from `markdex list`")
    remove_parser.set_defaults(func=cmd_remove)

    args = parser.parse_args()
    args.func(args)
