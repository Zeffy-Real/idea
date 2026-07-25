"""Unit tests for core data models (Document, Chunk, Message, SearchResult, Citation, etc.)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from cognita.core.models import (
    Chunk,
    Citation,
    Document,
    DocumentStatus,
    GenerationResponse,
    IngestionResult,
    Message,
    SearchResult,
)


# --------------------------------------------------------------------------- #
# Document
# --------------------------------------------------------------------------- #


class TestDocument:
    """Tests for the Document model."""

    def test_document_creation_basic(self) -> None:
        """A Document can be created with required fields."""
        doc = Document(title="Test Doc", source="/path/to/file.txt", content="Hello world")
        assert doc.title == "Test Doc"
        assert doc.source == "/path/to/file.txt"
        assert doc.content == "Hello world"
        assert doc.status == DocumentStatus.PENDING
        assert doc.chunk_ids == []
        assert doc.error is None
        assert doc.id  # auto-generated UUID hex
        assert isinstance(doc.created_at, datetime)
        assert isinstance(doc.updated_at, datetime)

    def test_document_default_file_type(self) -> None:
        """file_type defaults to an empty string."""
        doc = Document(title="T", source="s", content="c")
        assert doc.file_type == ""

    def test_document_metadata_defaults_to_empty_dict(self) -> None:
        """metadata defaults to an empty dict and is per-instance."""
        doc1 = Document(title="A", source="s", content="c")
        doc2 = Document(title="B", source="s", content="c")
        doc1.metadata["key"] = "value"
        assert "key" not in doc2.metadata

    def test_document_from_file(self, tmp_path: Path) -> None:
        """from_file creates a Document from a file path and content."""
        file_path = tmp_path / "report.md"
        file_path.write_text("# Report\n\nContent here.", encoding="utf-8")

        doc = Document.from_file(path=file_path, content="# Report\n\nContent here.")
        assert doc.title == "report"  # stem of the file name
        assert doc.file_type == "md"
        assert doc.content == "# Report\n\nContent here."
        assert doc.source == str(file_path.resolve())
        assert doc.file_size > 0

    def test_document_from_file_with_kwargs(self, tmp_path: Path) -> None:
        """from_file accepts extra kwargs (e.g. custom title, metadata)."""
        file_path = tmp_path / "data.txt"
        file_path.write_text("data", encoding="utf-8")

        doc = Document.from_file(
            path=file_path,
            content="data",
            title="Custom Title",
            metadata={"author": "tester"},
        )
        assert doc.title == "Custom Title"
        assert doc.metadata["author"] == "tester"

    def test_document_from_file_nonexistent_path(self) -> None:
        """from_file handles a path that does not exist on disk."""
        doc = Document.from_file(path="/nonexistent/file.txt", content="content")
        assert doc.title == "file"
        assert doc.file_type == "txt"
        # file_size falls back to byte length of content
        assert doc.file_size == len("content".encode("utf-8"))


# --------------------------------------------------------------------------- #
# Chunk
# --------------------------------------------------------------------------- #


class TestChunk:
    """Tests for the Chunk model."""

    def test_chunk_creation_basic(self) -> None:
        """A Chunk can be created with required fields."""
        chunk = Chunk(document_id="doc-1", content="Some text content")
        assert chunk.document_id == "doc-1"
        assert chunk.content == "Some text content"
        assert chunk.index == 0
        assert chunk.token_count == 0
        assert chunk.metadata == {}
        assert chunk.embedding is None
        assert chunk.id  # auto-generated

    def test_chunk_with_embedding_is_none_by_default(self) -> None:
        """embedding is None by default."""
        chunk = Chunk(document_id="d", content="c")
        assert chunk.embedding is None

    def test_chunk_with_embedding_returns_copy(self) -> None:
        """with_embedding returns a new Chunk with the embedding set."""
        chunk = Chunk(document_id="d", content="c")
        embedding = [0.1, 0.2, 0.3]
        new_chunk = chunk.with_embedding(embedding)
        assert new_chunk.embedding == embedding
        # Original is unchanged
        assert chunk.embedding is None
        # Other fields are preserved
        assert new_chunk.id == chunk.id
        assert new_chunk.document_id == chunk.document_id
        assert new_chunk.content == chunk.content

    def test_chunk_metadata_is_per_instance(self) -> None:
        """metadata is a fresh dict per instance."""
        c1 = Chunk(document_id="d", content="c")
        c2 = Chunk(document_id="d", content="c")
        c1.metadata["k"] = "v"
        assert "k" not in c2.metadata


# --------------------------------------------------------------------------- #
# Citation
# --------------------------------------------------------------------------- #


class TestCitation:
    """Tests for the Citation model and from_search_result."""

    def test_citation_from_search_result(self) -> None:
        """from_search_result creates a Citation from a SearchResult."""
        chunk = Chunk(
            id="chunk-001",
            document_id="doc-001",
            content="A" * 600,  # longer than 500 chars to test snippet truncation
            index=2,
            token_count=100,
            metadata={"title": "Source Doc"},
        )
        result = SearchResult(
            chunk=chunk,
            score=0.88,
            source_title="Source Doc",
            source_path="/path/to/source.txt",
        )
        citation = Citation.from_search_result(result)
        assert citation.chunk_id == "chunk-001"
        assert citation.document_id == "doc-001"
        assert citation.document_title == "Source Doc"
        assert citation.source == "/path/to/source.txt"
        assert citation.score == 0.88
        assert citation.chunk_index == 2
        # Snippet is truncated to 500 chars
        assert len(citation.content_snippet) == 500

    def test_citation_from_search_result_short_content(self) -> None:
        """from_search_result keeps full content when shorter than 500 chars."""
        chunk = Chunk(
            id="c1",
            document_id="d1",
            content="Short content.",
            index=0,
        )
        result = SearchResult(chunk=chunk, score=0.5, source_title="T", source_path="P")
        citation = Citation.from_search_result(result)
        assert citation.content_snippet == "Short content."


# --------------------------------------------------------------------------- #
# DocumentStatus enum
# --------------------------------------------------------------------------- #


class TestDocumentStatus:
    """Tests for the DocumentStatus enum."""

    def test_enum_values(self) -> None:
        """All expected enum members exist with correct string values."""
        assert DocumentStatus.PENDING == "pending"
        assert DocumentStatus.LOADING == "loading"
        assert DocumentStatus.CHUNKING == "chunking"
        assert DocumentStatus.EMBEDDING == "embedding"
        assert DocumentStatus.INDEXED == "indexed"
        assert DocumentStatus.FAILED == "failed"

    def test_enum_from_string(self) -> None:
        """DocumentStatus can be constructed from a string value."""
        assert DocumentStatus("pending") is DocumentStatus.PENDING
        assert DocumentStatus("failed") is DocumentStatus.FAILED

    def test_enum_is_str(self) -> None:
        """DocumentStatus members are also strings."""
        assert isinstance(DocumentStatus.PENDING, str)


# --------------------------------------------------------------------------- #
# GenerationResponse
# --------------------------------------------------------------------------- #


class TestGenerationResponse:
    """Tests for the GenerationResponse model."""

    def test_creation_with_defaults(self) -> None:
        """GenerationResponse can be created with only the required answer field."""
        resp = GenerationResponse(answer="The answer is 42.")
        assert resp.answer == "The answer is 42."
        assert resp.citations == []
        assert resp.usage == {}
        assert resp.model == ""
        assert resp.thinking is None
        assert resp.latency_ms == 0.0

    def test_creation_with_all_fields(self) -> None:
        """GenerationResponse can be created with all fields populated."""
        chunk = Chunk(id="c1", document_id="d1", content="content", index=0)
        result = SearchResult(chunk=chunk, score=0.9, source_title="T", source_path="P")
        citation = Citation.from_search_result(result)

        resp = GenerationResponse(
            answer="Answer with citation [1].",
            citations=[citation],
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            model="deepseek-chat",
            thinking="Reasoning trace...",
            latency_ms=42.5,
        )
        assert len(resp.citations) == 1
        assert resp.usage["total_tokens"] == 150
        assert resp.model == "deepseek-chat"
        assert resp.thinking == "Reasoning trace..."
        assert resp.latency_ms == 42.5


# --------------------------------------------------------------------------- #
# IngestionResult
# --------------------------------------------------------------------------- #


class TestIngestionResult:
    """Tests for the IngestionResult model."""

    def test_creation_success(self) -> None:
        """IngestionResult can represent a successful ingestion."""
        result = IngestionResult(
            document_id="doc-001",
            title="Test Document",
            chunks_created=5,
            chunks_indexed=5,
            status=DocumentStatus.INDEXED,
            latency_ms=123.45,
        )
        assert result.document_id == "doc-001"
        assert result.title == "Test Document"
        assert result.chunks_created == 5
        assert result.chunks_indexed == 5
        assert result.status == DocumentStatus.INDEXED
        assert result.error is None
        assert result.latency_ms == 123.45

    def test_creation_failure(self) -> None:
        """IngestionResult can represent a failed ingestion."""
        result = IngestionResult(
            document_id="doc-002",
            title="Bad Doc",
            chunks_created=0,
            chunks_indexed=0,
            status=DocumentStatus.FAILED,
            error="File not found",
            latency_ms=5.0,
        )
        assert result.status == DocumentStatus.FAILED
        assert result.error == "File not found"


# --------------------------------------------------------------------------- #
# Message
# --------------------------------------------------------------------------- #


class TestMessage:
    """Tests for the Message model."""

    def test_creation_basic(self) -> None:
        """A Message can be created with role and content."""
        msg = Message(role="user", content="What is AI?")
        assert msg.role == "user"
        assert msg.content == "What is AI?"
        assert msg.metadata == {}

    def test_creation_with_metadata(self) -> None:
        """A Message can carry metadata."""
        msg = Message(
            role="assistant",
            content="AI is ...",
            metadata={"citations": [{"chunk_id": "c1"}]},
        )
        assert msg.metadata["citations"] == [{"chunk_id": "c1"}]

    def test_metadata_is_per_instance(self) -> None:
        """metadata is a fresh dict per instance."""
        m1 = Message(role="user", content="a")
        m2 = Message(role="user", content="b")
        m1.metadata["k"] = "v"
        assert "k" not in m2.metadata

    def test_all_roles(self) -> None:
        """Message accepts arbitrary role strings."""
        for role in ("system", "user", "assistant"):
            msg = Message(role=role, content="content")
            assert msg.role == role
