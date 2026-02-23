#!/usr/bin/env python3
"""Index Markdown files into ChromaDB for markdex retrieval."""

import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from markdex.chunker import chunk_markdown
from markdex.config import CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL, ensure_data_dir


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


def list_indexed(
    chroma_path: str = CHROMA_DB_PATH,
    collection_name: str = COLLECTION_NAME,
) -> list[tuple[str, int]]:
    """Return indexed files as a sorted list of (file_path, chunk_count) tuples.

    The list order is stable (sorted by file path) so callers can use
    1-based indices for user-facing numbering.
    """
    ensure_data_dir()
    db = chromadb.PersistentClient(path=chroma_path)
    collection = db.get_or_create_collection(
        collection_name, metadata={"hnsw:space": "cosine"}
    )

    results = collection.get(include=["metadatas"])
    if not results["metadatas"]:
        return []

    files: dict[str, int] = {}
    for meta in results["metadatas"]:
        fp = meta.get("file_path", "unknown")
        files[fp] = files.get(fp, 0) + 1

    return sorted(files.items())


def remove_from_index(
    file_path: str,
    chroma_path: str = CHROMA_DB_PATH,
    collection_name: str = COLLECTION_NAME,
) -> int:
    """Remove all chunks for a given file_path from the index.

    Returns the number of chunks deleted.
    """
    ensure_data_dir()
    db = chromadb.PersistentClient(path=chroma_path)
    collection = db.get_or_create_collection(
        collection_name, metadata={"hnsw:space": "cosine"}
    )

    results = collection.get(
        where={"file_path": file_path},
        include=[],
    )
    ids_to_delete = results["ids"]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
    return len(ids_to_delete)
