"""Integration tests for the indexer (requires chromadb + sentence-transformers)."""

import shutil
from pathlib import Path

import chromadb
import chromadb.api.client
import pytest

from markdex.indexer import index_documents

FIXTURES_DIR = str(Path(__file__).parent / "fixtures" / "sample_docs")
TEST_DB_PATH = str(Path(__file__).parent / "fixtures" / "test_chroma_db")
TEST_COLLECTION = "test_markdown_docs"


@pytest.fixture(autouse=True)
def clean_test_db():
    """Remove test ChromaDB before and after each test."""
    chromadb.api.client.Client.clear_system_cache()
    shutil.rmtree(TEST_DB_PATH, ignore_errors=True)
    yield
    chromadb.api.client.Client.clear_system_cache()
    shutil.rmtree(TEST_DB_PATH, ignore_errors=True)


class TestIndexer:
    def test_indexes_all_files(self):
        stats = index_documents(
            paths=[FIXTURES_DIR],
            chroma_path=TEST_DB_PATH,
            collection_name=TEST_COLLECTION,
        )
        assert stats["files"] == 2
        assert stats["chunks"] > 0

    def test_chunks_stored_in_chromadb(self):
        index_documents(
            paths=[FIXTURES_DIR],
            chroma_path=TEST_DB_PATH,
            collection_name=TEST_COLLECTION,
        )
        db = chromadb.PersistentClient(path=TEST_DB_PATH)
        collection = db.get_or_create_collection(TEST_COLLECTION)
        assert collection.count() > 0

    def test_metadata_stored(self):
        index_documents(
            paths=[FIXTURES_DIR],
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
            paths=[FIXTURES_DIR],
            chroma_path=TEST_DB_PATH,
            collection_name=TEST_COLLECTION,
        )
        db = chromadb.PersistentClient(path=TEST_DB_PATH)
        collection = db.get_or_create_collection(TEST_COLLECTION)
        first_count = collection.count()

        # Index again with reset
        stats = index_documents(
            paths=[FIXTURES_DIR],
            chroma_path=TEST_DB_PATH,
            collection_name=TEST_COLLECTION,
            reset=True,
        )
        db2 = chromadb.PersistentClient(path=TEST_DB_PATH)
        collection2 = db2.get_or_create_collection(TEST_COLLECTION)
        assert collection2.count() == first_count  # same files, same count
        assert stats["files"] == 2
