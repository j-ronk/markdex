"""Markdown chunking: split files into semantically meaningful chunks by headings."""

import hashlib
import re
from dataclasses import dataclass, field

MAX_CHUNK_CHARS = 1000
MIN_CHUNK_CHARS = 50
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
IMAGE_PATTERN = re.compile(r"!\[.*?\]\((.+?)\)")


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


def chunk_markdown(content: str, file_path: str) -> list[Chunk]:
    """Split markdown content into chunks based on headings.

    Each heading starts a new chunk. Sections >1000 chars are split at paragraph
    boundaries. Sections <50 chars are merged with the following section.
    """
    content = content.strip()
    if not content:
        return []

    file_hash = hashlib.md5(content.encode()).hexdigest()
    matches = list(HEADING_PATTERN.finditer(content))

    if not matches:
        return [_make_chunk(content, file_path, "", 0, file_hash)]

    raw_chunks: list[Chunk] = []

    # Content before first heading
    if matches[0].start() > 0:
        pre = content[: matches[0].start()].strip()
        if pre:
            raw_chunks.append(_make_chunk(pre, file_path, "", 0, file_hash))

    # Each heading section
    for i, match in enumerate(matches):
        level = len(match.group(1))
        heading = match.group(2).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        section = content[start:end].strip()

        if len(section) > MAX_CHUNK_CHARS:
            raw_chunks.extend(_split_large(section, file_path, heading, level, file_hash))
        else:
            raw_chunks.append(_make_chunk(section, file_path, heading, level, file_hash))

    return _merge_small(raw_chunks)


def _make_chunk(text: str, file_path: str, heading: str, level: int, file_hash: str) -> Chunk:
    return Chunk(
        text=text,
        metadata={
            "file_path": file_path,
            "heading": heading,
            "heading_level": level,
            "image_refs": IMAGE_PATTERN.findall(text),
            "file_hash": file_hash,
        },
    )


def _split_large(text: str, file_path: str, heading: str, level: int, file_hash: str) -> list[Chunk]:
    """Split a large section at paragraph boundaries (double newline)."""
    paragraphs = re.split(r"\n\n+", text)
    chunks: list[Chunk] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) > MAX_CHUNK_CHARS and current:
            chunks.append(_make_chunk(current.strip(), file_path, heading, level, file_hash))
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para

    if current.strip():
        chunks.append(_make_chunk(current.strip(), file_path, heading, level, file_hash))

    return chunks


def _merge_small(chunks: list[Chunk]) -> list[Chunk]:
    """Merge chunks smaller than MIN_CHUNK_CHARS with the next chunk."""
    if len(chunks) <= 1:
        return chunks

    merged: list[Chunk] = []
    i = 0
    while i < len(chunks):
        if len(chunks[i].text) < MIN_CHUNK_CHARS and i + 1 < len(chunks):
            combined_text = chunks[i].text + "\n\n" + chunks[i + 1].text
            merged.append(Chunk(text=combined_text, metadata=chunks[i + 1].metadata))
            i += 2
        else:
            merged.append(chunks[i])
            i += 1

    return merged
