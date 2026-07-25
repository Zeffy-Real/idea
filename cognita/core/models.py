"""
Shared data models used across the application.
These are the fundamental data structures that flow through the pipeline:
  Document -> Chunk -> Embedding -> VectorStore -> SearchResult -> Citation
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _generate_id() -> str:
    return uuid.uuid4().hex


class DocumentStatus(str, Enum):
    """Lifecycle status of a document in the system."""

    PENDING = "pending"
    LOADING = "loading"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXED = "indexed"
    FAILED = "failed"


class Document(BaseModel):
    """A source document loaded from disk or uploaded by a user."""

    id: str = Field(default_factory=_generate_id)
    title: str
    source: str  # File path or URL
    content: str
    file_type: str = ""  # pdf, md, txt, docx
    file_size: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: DocumentStatus = DocumentStatus.PENDING
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
    chunk_ids: list[str] = Field(default_factory=list)
    error: str | None = None

    @classmethod
    def from_file(cls, path: str | Path, content: str, **kwargs: Any) -> Document:
        """Create a Document from a file path."""
        p = Path(path)
        return cls(
            title=kwargs.pop("title", p.stem),
            source=str(p.resolve()),
            content=content,
            file_type=p.suffix.lstrip(".").lower(),
            file_size=p.stat().st_size if p.exists() else len(content.encode()),
            **kwargs,
        )


class Chunk(BaseModel):
    """A text chunk extracted from a document, ready for embedding."""

    id: str = Field(default_factory=_generate_id)
    document_id: str
    content: str
    index: int = 0  # Position within the source document
    token_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Populated after embedding
    embedding: list[float] | None = None

    def with_embedding(self, embedding: list[float]) -> Chunk:
        """Return a copy with the embedding set."""
        return self.model_copy(update={"embedding": embedding})


class Message(BaseModel):
    """A chat message in a conversation."""

    role: str  # system, user, assistant
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """A single retrieval result from the vector store."""

    chunk: Chunk
    score: float
    source_title: str = ""
    source_path: str = ""


class Citation(BaseModel):
    """A citation referencing a source document chunk."""

    chunk_id: str
    document_id: str
    document_title: str
    source: str
    content_snippet: str
    score: float
    chunk_index: int = 0

    @classmethod
    def from_search_result(cls, result: SearchResult) -> Citation:
        """Create a citation from a search result."""
        return cls(
            chunk_id=result.chunk.id,
            document_id=result.chunk.document_id,
            document_title=result.source_title,
            source=result.source_path,
            content_snippet=result.chunk.content[:500],
            score=result.score,
            chunk_index=result.chunk.index,
        )


class GenerationResponse(BaseModel):
    """The complete response from the RAG generation pipeline."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    model: str = ""
    thinking: str | None = None  # Reasoning trace (if thinking mode enabled)
    latency_ms: float = 0.0


class IngestionResult(BaseModel):
    """Result of ingesting a document into the knowledge base."""

    document_id: str
    title: str
    chunks_created: int
    chunks_indexed: int
    status: DocumentStatus
    error: str | None = None
    latency_ms: float = 0.0
