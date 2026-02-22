"""Centralized configuration for markdex."""

from pathlib import Path

DATA_DIR = Path.home() / ".markdex"
CHROMA_DB_PATH = str(DATA_DIR / "chroma_db")
COLLECTION_NAME = "markdown_docs"
EMBEDDING_MODEL = "all-mpnet-base-v2"


def ensure_data_dir() -> None:
    """Create the data directory if it doesn't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
