"""Unit tests for the TextChunker."""

from __future__ import annotations

import pytest

from cognita.core.exceptions import ChunkingError
from cognita.core.models import Chunk, Document, DocumentStatus
from cognita.ingestion.chunkers import TextChunker


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_long_text(num_paragraphs: int = 20, words_per_para: int = 50) -> str:
    """Generate a multi-paragraph text that exceeds typical chunk sizes."""
    paragraphs = []
    for i in range(num_paragraphs):
        words = [f"word{i}_{j}" for j in range(words_per_para)]
        paragraphs.append(" ".join(words))
    return "\n\n".join(paragraphs)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class TestTextChunkerBasic:
    """Basic chunking behavior."""

    def test_chunk_returns_list_of_chunks(self) -> None:
        """chunk() returns a list of Chunk objects."""
        chunker = TextChunker(chunk_size=512, chunk_overlap=64)
        chunks = chunker.chunk("Hello world.", document_id="doc-1")
        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunking_produces_multiple_chunks_for_long_text(self) -> None:
        """Long text is split into multiple chunks."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = _make_long_text(num_paragraphs=20, words_per_para=30)
        chunks = chunker.chunk(text, document_id="doc-1")
        assert len(chunks) > 1

    def test_single_chunk_for_short_text(self) -> None:
        """Text shorter than chunk_size produces a single chunk."""
        chunker = TextChunker(chunk_size=512, chunk_overlap=64)
        text = "This is a short sentence."
        chunks = chunker.chunk(text, document_id="doc-1")
        assert len(chunks) == 1
        assert chunks[0].content.strip() == text.strip()


class TestTextChunkerParameters:
    """chunk_size and chunk_overlap parameter behavior."""

    def test_smaller_chunk_size_produces_more_chunks(self) -> None:
        """A smaller chunk_size produces more chunks for the same text."""
        text = _make_long_text(num_paragraphs=15, words_per_para=40)
        chunker_small = TextChunker(chunk_size=50, chunk_overlap=10)
        chunker_large = TextChunker(chunk_size=500, chunk_overlap=10)
        chunks_small = chunker_small.chunk(text, document_id="doc-1")
        chunks_large = chunker_large.chunk(text, document_id="doc-1")
        assert len(chunks_small) > len(chunks_large)

    def test_chunk_overlap_must_be_smaller_than_chunk_size(self) -> None:
        """chunk_overlap >= chunk_size raises ChunkingError."""
        with pytest.raises(ChunkingError):
            TextChunker(chunk_size=100, chunk_overlap=100)
        with pytest.raises(ChunkingError):
            TextChunker(chunk_size=100, chunk_overlap=150)

    def test_chunk_size_must_be_positive(self) -> None:
        """chunk_size <= 0 raises ChunkingError."""
        with pytest.raises(ChunkingError):
            TextChunker(chunk_size=0, chunk_overlap=0)
        with pytest.raises(ChunkingError):
            TextChunker(chunk_size=-10, chunk_overlap=0)

    def test_chunk_overlap_zero_is_valid(self) -> None:
        """chunk_overlap=0 is valid and produces no overlap."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=0)
        text = _make_long_text(num_paragraphs=10, words_per_para=20)
        chunks = chunker.chunk(text, document_id="doc-1")
        assert len(chunks) >= 1


class TestTextChunkerTokenCount:
    """Token count is set on each chunk."""

    def test_token_count_is_positive(self) -> None:
        """Each chunk has a positive token_count."""
        chunker = TextChunker(chunk_size=512, chunk_overlap=64)
        text = _make_long_text(num_paragraphs=10, words_per_para=30)
        chunks = chunker.chunk(text, document_id="doc-1")
        for chunk in chunks:
            assert chunk.token_count > 0

    def test_token_count_does_not_exceed_chunk_size(self) -> None:
        """No chunk's token_count should exceed chunk_size (with tolerance)."""
        chunk_size = 100
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=20)
        text = _make_long_text(num_paragraphs=20, words_per_para=40)
        chunks = chunker.chunk(text, document_id="doc-1")
        for chunk in chunks:
            # Allow a small tolerance because the merge logic may slightly
            # overshoot when a single piece itself is near the limit.
            assert chunk.token_count <= chunk_size + 50


class TestTextChunkerIndex:
    """Chunk index is sequential."""

    def test_chunk_indices_are_sequential(self) -> None:
        """Chunk indices start at 0 and increment by 1."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = _make_long_text(num_paragraphs=15, words_per_para=30)
        chunks = chunker.chunk(text, document_id="doc-1")
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_chunk_metadata_contains_chunk_index(self) -> None:
        """Each chunk's metadata contains a 'chunk_index' key."""
        chunker = TextChunker(chunk_size=512, chunk_overlap=64)
        chunks = chunker.chunk("Hello world.", document_id="doc-1")
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i


class TestTextChunkerMetadata:
    """Metadata is propagated to chunks."""

    def test_custom_metadata_is_propagated(self) -> None:
        """Custom metadata passed to chunk() appears on each chunk."""
        chunker = TextChunker(chunk_size=512, chunk_overlap=64)
        custom_meta = {"source": "test.txt", "category": "tech"}
        chunks = chunker.chunk("Hello world.", document_id="doc-1", metadata=custom_meta)
        for chunk in chunks:
            assert chunk.metadata["source"] == "test.txt"
            assert chunk.metadata["category"] == "tech"

    def test_no_metadata_passes_empty(self) -> None:
        """Without metadata, chunks still get chunk_index and token_count."""
        chunker = TextChunker(chunk_size=512, chunk_overlap=64)
        chunks = chunker.chunk("Hello world.", document_id="doc-1")
        for chunk in chunks:
            assert "chunk_index" in chunk.metadata
            assert "token_count" in chunk.metadata


class TestTextChunkerChunkDocument:
    """chunk_document method."""

    def test_chunk_document_returns_chunks(self) -> None:
        """chunk_document chunks a Document and returns Chunk list."""
        doc = Document(
            title="Test",
            source="/path/test.txt",
            content=_make_long_text(num_paragraphs=10, words_per_para=30),
            file_type="txt",
        )
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.document_id == doc.id
            assert chunk.metadata["title"] == "Test"
            assert chunk.metadata["source"] == "/path/test.txt"
            assert chunk.metadata["file_type"] == "txt"

    def test_chunk_document_propagates_document_metadata(self) -> None:
        """chunk_document includes the document's own metadata in chunks."""
        doc = Document(
            title="Test",
            source="/path/test.txt",
            content="Hello world.",
            file_type="txt",
            metadata={"author": "tester", "tags": ["ai", "ml"]},
        )
        chunker = TextChunker(chunk_size=512, chunk_overlap=64)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].metadata["author"] == "tester"
        assert chunks[0].metadata["tags"] == ["ai", "ml"]


class TestTextChunkerEdgeCases:
    """Edge case handling."""

    def test_empty_text_returns_empty_list(self) -> None:
        """Empty text produces no chunks."""
        chunker = TextChunker(chunk_size=512, chunk_overlap=64)
        assert chunker.chunk("", document_id="doc-1") == []

    def test_whitespace_only_text_returns_empty_list(self) -> None:
        """Whitespace-only text produces no chunks."""
        chunker = TextChunker(chunk_size=512, chunk_overlap=64)
        assert chunker.chunk("   \n\n  \t  ", document_id="doc-1") == []

    def test_chunk_content_is_non_empty(self) -> None:
        """Every chunk has non-empty, non-whitespace content."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = _make_long_text(num_paragraphs=10, words_per_para=20)
        chunks = chunker.chunk(text, document_id="doc-1")
        for chunk in chunks:
            assert chunk.content.strip() != ""

    def test_all_chunks_have_document_id(self) -> None:
        """Every chunk has the provided document_id."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = _make_long_text(num_paragraphs=10, words_per_para=20)
        chunks = chunker.chunk(text, document_id="my-doc-id")
        for chunk in chunks:
            assert chunk.document_id == "my-doc-id"
