"""Pydantic schemas for API request and response validation.

These models define the contract between the API layer and its clients.
They map to (but are intentionally decoupled from) the core domain models
in :mod:`cognita.core.models`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from cognita.core.models import Citation


class QueryRequest(BaseModel):
    """Request body for the query endpoint."""

    query: str = Field(..., min_length=1, description="The user's question")
    top_k: int | None = Field(
        default=None,
        ge=1,
        description="Number of documents to retrieve. Defaults to settings.",
    )
    score_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score. Defaults to settings.",
    )
    thinking: bool | None = Field(
        default=None,
        description="Enable thinking/reasoning mode. Defaults to settings.",
    )
    conversation_id: str | None = Field(
        default=None,
        description="Identifier for multi-turn conversation memory.",
    )
    filter_conditions: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata filters forwarded to the vector store.",
    )
    expand_query: bool = Field(
        default=False,
        description="Whether to expand the query using LLM-generated variations.",
    )


class QueryResponse(BaseModel):
    """Response body for the query endpoint."""

    answer: str = Field(..., description="The generated answer")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Source citations supporting the answer",
    )
    usage: dict[str, Any] = Field(
        default_factory=dict,
        description="Token usage statistics",
    )
    model: str = Field(..., description="The LLM model used for generation")
    thinking: str | None = Field(
        default=None,
        description="Reasoning trace if thinking mode was enabled",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="End-to-end generation latency in milliseconds",
    )


class TextIngestRequest(BaseModel):
    """Request body for the text ingestion endpoint."""

    title: str = Field(..., min_length=1, description="Document title")
    content: str = Field(..., min_length=1, description="Document text content")
    source: str = Field(
        default="user_input",
        description="Source identifier for the document",
    )


class DocumentResponse(BaseModel):
    """Response body for document ingestion endpoints."""

    document_id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    chunks_created: int = Field(..., ge=0, description="Number of chunks created")
    chunks_indexed: int = Field(..., ge=0, description="Number of chunks indexed")
    status: str = Field(..., description="Ingestion status")
    error: str | None = Field(default=None, description="Error message if failed")
    latency_ms: float = Field(..., ge=0.0, description="Ingestion latency in ms")


class HealthResponse(BaseModel):
    """Response body for the health check endpoint."""

    status: str = Field(..., description="Overall service status")
    version: str = Field(..., description="Application version")
    components: dict[str, bool] = Field(
        default_factory=dict,
        description="Health status of individual components",
    )


class ErrorResponse(BaseModel):
    """Standard error response body."""

    error: str = Field(..., description="Error message")
    details: dict[str, Any] | None = Field(
        default=None,
        description="Additional error details",
    )


class DocumentListResponse(BaseModel):
    """Response body for the document listing endpoint."""

    total_chunks: int = Field(..., ge=0, description="Total number of chunks in the store")
    components: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional document statistics",
    )
