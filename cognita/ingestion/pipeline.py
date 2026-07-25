"""
End-to-end ingestion pipeline.

The :class:`IngestionPipeline` orchestrates the four stages of document
ingestion:

    load  ->  chunk  ->  embed  ->  index

Each stage updates the :class:`Document`'s status so callers can observe
progress. Failures at any stage are caught, logged, recorded as metrics,
and surfaced as a :class:`IngestionResult` with ``status=FAILED`` rather
than raising — this lets batch ingestion (``ingest_directory``) continue
processing remaining files.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cognita.config import get_settings
from cognita.core.embedding import LocalEmbedding, get_embedding
from cognita.core.exceptions import CognitaError, DocumentLoadingError
from cognita.core.models import Chunk, Document, DocumentStatus, IngestionResult
from cognita.core.vectorstore import BaseVectorStore, get_vectorstore
from cognita.ingestion.chunkers import TextChunker
from cognita.ingestion.loaders import DocumentLoader
from cognita.observability.logging import get_logger
from cognita.observability.metrics import (
    ingestion_chunks_created,
    ingestion_duration,
    ingestion_requests_total,
)

logger = get_logger("cognita.ingestion.pipeline")

# Number of chunks embedded per batch call.
_EMBED_BATCH_SIZE = 32


class IngestionPipeline:
    """Orchestrates loading, chunking, embedding, and indexing of documents."""

    def __init__(
        self,
        loader: DocumentLoader | None = None,
        chunker: TextChunker | None = None,
        embedding: LocalEmbedding | None = None,
        vectorstore: BaseVectorStore | None = None,
    ) -> None:
        self._loader = loader or DocumentLoader()

        if chunker is None:
            settings = get_settings()
            chunker = TextChunker(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
        self._chunker = chunker

        self._embedding = embedding or get_embedding()
        self._vectorstore = vectorstore or get_vectorstore()
        self._logger = get_logger("cognita.ingestion.pipeline")

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def initialize(self) -> None:
        """Create the vector-store collection using the embedding dimension."""
        dimension = self._embedding.dimension
        await self._vectorstore.create_collection(dimension)
        self._logger.info(
            "Ingestion pipeline initialised", embedding_dimension=dimension
        )

    # ------------------------------------------------------------------ #
    # Public ingestion entry-points
    # ------------------------------------------------------------------ #

    async def ingest_file(self, file_path: str | Path) -> IngestionResult:
        """Load and ingest a single file from disk."""
        path = Path(file_path)
        file_type = path.suffix.lstrip(".").lower()
        start = time.perf_counter()

        # --- Stage 1: load -------------------------------------------------
        try:
            document = await self._loader.load(path)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            self._logger.error(
                "Document loading failed",
                path=str(path),
                error=str(exc),
                exc_info=True,
            )
            ingestion_requests_total.labels(
                file_type=file_type, status="error"
            ).inc()
            ingestion_duration.labels(file_type=file_type).observe(elapsed)
            return IngestionResult(
                document_id="",
                title=path.stem,
                chunks_created=0,
                chunks_indexed=0,
                status=DocumentStatus.FAILED,
                error=str(exc),
                latency_ms=round(elapsed * 1000, 2),
            )

        return await self._ingest_document(document, file_type, start)

    async def ingest_text(
        self,
        title: str,
        content: str,
        source: str = "user_input",
    ) -> IngestionResult:
        """Ingest a raw text string without a backing file."""
        document = Document(
            title=title,
            source=source,
            content=content,
            file_type="txt",
            file_size=len(content.encode("utf-8")),
        )
        start = time.perf_counter()
        return await self._ingest_document(document, "txt", start)

    async def ingest_directory(
        self, dir_path: str | Path
    ) -> list[IngestionResult]:
        """Ingest every supported file beneath *dir_path*."""
        path = Path(dir_path)
        self._logger.info("Ingesting directory", path=str(path))

        try:
            documents = await self._loader.load_directory(path)
        except DocumentLoadingError as exc:
            self._logger.error(
                "Failed to load directory", path=str(path), error=str(exc)
            )
            return []

        self._logger.info(
            "Loaded documents from directory",
            path=str(path),
            count=len(documents),
        )

        results: list[IngestionResult] = []
        for document in documents:
            result = await self._ingest_document(document, document.file_type)
            results.append(result)

        succeeded = sum(
            1 for r in results if r.status == DocumentStatus.INDEXED
        )
        self._logger.info(
            "Directory ingestion complete",
            path=str(path),
            total=len(results),
            succeeded=succeeded,
            failed=len(results) - succeeded,
        )
        return results

    # ------------------------------------------------------------------ #
    # Core processing (shared by all entry-points)
    # ------------------------------------------------------------------ #

    async def _ingest_document(
        self,
        document: Document,
        file_type: str,
        start: float | None = None,
    ) -> IngestionResult:
        """Run chunk -> embed -> index for a single document."""
        if start is None:
            start = time.perf_counter()

        try:
            # Mark as loading (document content is already available).
            self._update_status(document, DocumentStatus.LOADING)
            self._logger.info(
                "Ingesting document",
                document_id=document.id,
                title=document.title,
                file_type=file_type,
            )

            # --- Stage 2: chunk ----------------------------------------
            self._update_status(document, DocumentStatus.CHUNKING)
            chunks = self._chunker.chunk_document(document)
            document.chunk_ids = [c.id for c in chunks]
            self._logger.info(
                "Document chunked",
                document_id=document.id,
                chunks=len(chunks),
            )

            # --- Stage 3: embed ----------------------------------------
            self._update_status(document, DocumentStatus.EMBEDDING)
            embedded_chunks = await self._embed_chunks(chunks)
            self._logger.info(
                "Chunks embedded",
                document_id=document.id,
                count=len(embedded_chunks),
            )

            # --- Stage 4: index ----------------------------------------
            indexed = await self._vectorstore.add(embedded_chunks)
            self._update_status(document, DocumentStatus.INDEXED)
            self._logger.info(
                "Document indexed",
                document_id=document.id,
                chunks_indexed=indexed,
            )

            elapsed = time.perf_counter() - start
            ingestion_requests_total.labels(
                file_type=file_type, status="success"
            ).inc()
            ingestion_duration.labels(file_type=file_type).observe(elapsed)
            ingestion_chunks_created.inc(len(chunks))

            return IngestionResult(
                document_id=document.id,
                title=document.title,
                chunks_created=len(chunks),
                chunks_indexed=indexed,
                status=DocumentStatus.INDEXED,
                latency_ms=round(elapsed * 1000, 2),
            )

        except Exception as exc:
            elapsed = time.perf_counter() - start
            self._update_status(document, DocumentStatus.FAILED, error=str(exc))
            self._logger.error(
                "Ingestion failed",
                document_id=document.id,
                title=document.title,
                error=str(exc),
                exc_info=True,
            )
            ingestion_requests_total.labels(
                file_type=file_type, status="error"
            ).inc()
            ingestion_duration.labels(file_type=file_type).observe(elapsed)

            return IngestionResult(
                document_id=document.id,
                title=document.title,
                chunks_created=0,
                chunks_indexed=0,
                status=DocumentStatus.FAILED,
                error=str(exc),
                latency_ms=round(elapsed * 1000, 2),
            )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    async def _embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Embed chunks in batches of ``_EMBED_BATCH_SIZE``."""
        if not chunks:
            return []

        embedded: list[Chunk] = []
        for offset in range(0, len(chunks), _EMBED_BATCH_SIZE):
            batch = chunks[offset : offset + _EMBED_BATCH_SIZE]
            texts = [c.content for c in batch]
            embeddings = await self._embedding.embed_batch_async(texts)
            for chunk, emb in zip(batch, embeddings):
                embedded.append(chunk.with_embedding(emb))
            self._logger.debug(
                "Embedded batch",
                batch=offset // _EMBED_BATCH_SIZE,
                size=len(batch),
            )
        return embedded

    @staticmethod
    def _update_status(
        document: Document,
        status: DocumentStatus,
        error: str | None = None,
    ) -> None:
        """Transition a document's status and refresh ``updated_at``."""
        document.status = status
        document.updated_at = datetime.now(timezone.utc)
        if error is not None:
            document.error = error
        elif status != DocumentStatus.FAILED:
            # Clear any stale error when moving to a non-failed state.
            document.error = None
