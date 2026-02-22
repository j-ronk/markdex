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


@pytest.fixture(scope="module")
def monkeypatch_module():
    """Module-scoped monkeypatch."""
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


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
