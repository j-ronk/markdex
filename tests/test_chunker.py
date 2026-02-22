"""Tests for Markdown chunking logic."""

from markdex.chunker import chunk_markdown, Chunk


class TestChunkMarkdownBasic:
    """Test basic heading-based splitting."""

    def test_single_heading_with_content(self):
        content = "# Introduction\n\nThis is the intro paragraph with enough detail to exceed the minimum chunk size."
        chunks = chunk_markdown(content, "test.md")
        assert len(chunks) == 1
        assert chunks[0].metadata["heading"] == "Introduction"
        assert chunks[0].metadata["heading_level"] == 1
        assert chunks[0].metadata["file_path"] == "test.md"
        assert "intro paragraph" in chunks[0].text

    def test_multiple_headings(self):
        content = "# Section A\n\nContent A has enough text to exceed the minimum chunk size threshold.\n\n## Section B\n\nContent B has enough text to exceed the minimum chunk size threshold."
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
        content = "Preamble text here that is long enough to exceed the minimum chunk size.\n\n# First Heading\n\nHeading content that is long enough to exceed the minimum chunk size."
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
        content = "# Title\n\nBody text that is long enough to exceed the minimum chunk size threshold."
        chunks = chunk_markdown(content, "docs/finance/q4.md")
        assert chunks[0].metadata["file_path"] == "docs/finance/q4.md"

    def test_file_hash_computed(self):
        content = "# Title\n\nBody text that is long enough to exceed the minimum chunk size threshold."
        chunks = chunk_markdown(content, "test.md")
        assert len(chunks[0].metadata["file_hash"]) == 32  # MD5 hex length

    def test_file_hash_changes_with_content(self):
        chunks_a = chunk_markdown("# A\n\nContent A that is long enough to exceed the minimum chunk size.", "test.md")
        chunks_b = chunk_markdown("# B\n\nContent B that is long enough to exceed the minimum chunk size.", "test.md")
        assert chunks_a[0].metadata["file_hash"] != chunks_b[0].metadata["file_hash"]

    def test_image_refs_extracted(self):
        content = "# Diagram\n\nSee below:\n\n![Chart](./images/chart.png)"
        chunks = chunk_markdown(content, "test.md")
        assert chunks[0].metadata["image_refs"] == ["./images/chart.png"]

    def test_multiple_image_refs(self):
        content = "# Images\n\n![A](a.png)\n\nDescriptive text between images.\n\n![B](b.png)"
        chunks = chunk_markdown(content, "test.md")
        assert chunks[0].metadata["image_refs"] == ["a.png", "b.png"]

    def test_no_image_refs(self):
        content = "# Plain\n\nNo images here, just plain text that exceeds the minimum chunk size."
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
