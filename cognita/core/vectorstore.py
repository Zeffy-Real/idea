"""
Vector store abstraction layer with Qdrant and in-memory implementations.
Provides document storage, similarity search, and collection management.
"""

from __future__ import annotations

import asyncio
import threading
import time
from abc import ABC, abstractmethod
from typing import Any

from cognita.config import Settings, get_settings
from cognita.core.exceptions import VectorStoreConnectionError, VectorStoreError
from cognita.core.models import Chunk, SearchResult
from cognita.observability.logging import get_logger
from cognita.observability.metrics import (
    vectorstore_collection_size,
    vectorstore_operations_total,
)

logger = get_logger(__name__)


class BaseVectorStore(ABC):
    """Abstract vector store interface."""

    @abstractmethod
    async def create_collection(self, dimension: int) -> None:
        """Create the collection if it doesn't exist."""

    @abstractmethod
    async def add(self, chunks: list[Chunk]) -> int:
        """Add chunks with embeddings to the store. Returns count added."""

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter_conditions: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar chunks."""

    @abstractmethod
    async def delete(self, chunk_ids: list[str]) -> int:
        """Delete chunks by ID. Returns count deleted."""

    @abstractmethod
    async def delete_by_document(self, document_id: str) -> int:
        """Delete all chunks for a document. Returns count deleted."""

    @abstractmethod
    async def count(self) -> int:
        """Get the total number of vectors in the collection."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the vector store is healthy."""


class QdrantVectorStore(BaseVectorStore):
    """Qdrant vector store implementation."""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._client: Any = None
        self._logger = get_logger("cognita.vectorstore.qdrant")
        self._collection_name = self._settings.qdrant_collection
        self._lock = threading.Lock()

    def _get_client(self) -> Any:
        """Lazy-initialize the Qdrant client."""
        if self._client is not None:
            return self._client

        with self._lock:
            if self._client is not None:
                return self._client

            try:
                from qdrant_client import AsyncQdrantClient

                client_kwargs: dict[str, Any] = {
                    "url": self._settings.qdrant_url,
                }
                if self._settings.qdrant_api_key:
                    client_kwargs["api_key"] = self._settings.qdrant_api_key

                self._client = AsyncQdrantClient(**client_kwargs, timeout=30)
                self._logger.info("Qdrant client initialized", url=self._settings.qdrant_url)

            except ImportError as e:
                raise VectorStoreError(
                    "qdrant-client not installed. Run: pip install qdrant-client",
                    store="qdrant",
                ) from e
            except Exception as e:
                raise VectorStoreConnectionError(
                    f"Failed to connect to Qdrant: {e}",
                    store="qdrant",
                ) from e

            return self._client

    async def create_collection(self, dimension: int) -> None:
        """Create the collection if it doesn't exist."""
        from qdrant_client.http.exceptions import UnexpectedResponse

        client = self._get_client()

        try:
            # Check if collection exists
            collections = await client.get_collections()
            exists = any(c.name == self._collection_name for c in collections.collections)

            if not exists:
                from qdrant_client.http.models import (
                    Distance,
                    VectorParams,
                )

                await client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(
                        size=dimension,
                        distance=Distance.COSINE,
                    ),
                )
                self._logger.info(
                    "Collection created",
                    collection=self._collection_name,
                    dimension=dimension,
                )
            else:
                self._logger.debug(
                    "Collection already exists",
                    collection=self._collection_name,
                )

        except UnexpectedResponse as e:
            vectorstore_operations_total.labels(operation="create_collection", status="error").inc()
            raise VectorStoreError(
                f"Failed to create collection: {e}",
                store="qdrant",
                operation="create_collection",
            ) from e

    async def add(self, chunks: list[Chunk]) -> int:
        """Add chunks with embeddings to Qdrant."""
        if not chunks:
            return 0

        # Filter chunks that have embeddings
        valid_chunks = [c for c in chunks if c.embedding is not None]
        if not valid_chunks:
            self._logger.warning("No chunks with embeddings to add")
            return 0

        from qdrant_client.http.models import PointStruct

        client = self._get_client()
        start = time.perf_counter()

        try:
            points = []
            for chunk in valid_chunks:
                points.append(
                    PointStruct(
                        id=chunk.id,
                        vector=chunk.embedding,
                        payload={
                            "content": chunk.content,
                            "document_id": chunk.document_id,
                            "chunk_index": chunk.index,
                            "token_count": chunk.token_count,
                            "title": chunk.metadata.get("title", ""),
                            "source": chunk.metadata.get("source", ""),
                            "file_type": chunk.metadata.get("file_type", ""),
                            **chunk.metadata,
                        },
                    )
                )

            # Batch upsert (Qdrant handles batching internally)
            await client.upsert(
                collection_name=self._collection_name,
                points=points,
                wait=True,
            )

            elapsed = time.perf_counter() - start
            vectorstore_operations_total.labels(operation="add", status="success").inc()

            # Update collection size gauge
            count = await self.count()
            vectorstore_collection_size.set(count)

            self._logger.info(
                "Chunks added to Qdrant",
                count=len(valid_chunks),
                latency_ms=round(elapsed * 1000, 2),
            )

            return len(valid_chunks)

        except Exception as e:
            vectorstore_operations_total.labels(operation="add", status="error").inc()
            self._logger.error("Failed to add chunks", error=str(e), count=len(valid_chunks))
            raise VectorStoreError(
                f"Failed to add chunks: {e}",
                store="qdrant",
                operation="add",
            ) from e

    async def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter_conditions: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar chunks in Qdrant."""
        from qdrant_client.http.models import FieldCondition, Filter, MatchValue

        client = self._get_client()
        start = time.perf_counter()

        # Build filter from conditions
        query_filter = None
        if filter_conditions:
            conditions = []
            for key, value in filter_conditions.items():
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            query_filter = Filter(must=conditions)

        try:
            results = await client.search(
                collection_name=self._collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True,
            )

            elapsed = time.perf_counter() - start
            vectorstore_operations_total.labels(operation="search", status="success").inc()

            search_results: list[SearchResult] = []
            for point in results:
                payload = point.payload or {}
                chunk = Chunk(
                    id=str(point.id),
                    document_id=payload.get("document_id", ""),
                    content=payload.get("content", ""),
                    index=payload.get("chunk_index", 0),
                    token_count=payload.get("token_count", 0),
                    metadata={k: v for k, v in payload.items() if k not in {
                        "content", "document_id", "chunk_index", "token_count",
                        "title", "source", "file_type",
                    }},
                )
                search_results.append(
                    SearchResult(
                        chunk=chunk,
                        score=point.score,
                        source_title=payload.get("title", ""),
                        source_path=payload.get("source", ""),
                    )
                )

            self._logger.debug(
                "Search completed",
                results=len(search_results),
                latency_ms=round(elapsed * 1000, 2),
            )

            return search_results

        except Exception as e:
            vectorstore_operations_total.labels(operation="search", status="error").inc()
            self._logger.error("Search failed", error=str(e))
            raise VectorStoreError(
                f"Search failed: {e}",
                store="qdrant",
                operation="search",
            ) from e

    async def delete(self, chunk_ids: list[str]) -> int:
        """Delete chunks by ID from Qdrant."""
        if not chunk_ids:
            return 0

        client = self._get_client()

        try:
            from qdrant_client.http.models import PointIdsList

            await client.delete(
                collection_name=self._collection_name,
                points_selector=PointIdsList(points=chunk_ids),
                wait=True,
            )

            vectorstore_operations_total.labels(operation="delete", status="success").inc()
            count = await self.count()
            vectorstore_collection_size.set(count)

            self._logger.info("Chunks deleted", count=len(chunk_ids))
            return len(chunk_ids)

        except Exception as e:
            vectorstore_operations_total.labels(operation="delete", status="error").inc()
            self._logger.error("Delete failed", error=str(e))
            raise VectorStoreError(
                f"Delete failed: {e}",
                store="qdrant",
                operation="delete",
            ) from e

    async def delete_by_document(self, document_id: str) -> int:
        """Delete all chunks for a document."""
        client = self._get_client()

        try:
            from qdrant_client.http.models import (
                Filter,
                FieldCondition,
                MatchValue,
            )

            # Count before delete
            count_result = await client.count(
                collection_name=self._collection_name,
                count_filter=Filter(
                    must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
                ),
                exact=True,
            )
            deleted_count = count_result.count

            await client.delete(
                collection_name=self._collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
                ),
                wait=True,
            )

            vectorstore_operations_total.labels(
                operation="delete_by_document", status="success"
            ).inc()

            count = await self.count()
            vectorstore_collection_size.set(count)

            self._logger.info(
                "Document chunks deleted",
                document_id=document_id,
                count=deleted_count,
            )
            return deleted_count

        except Exception as e:
            vectorstore_operations_total.labels(
                operation="delete_by_document", status="error"
            ).inc()
            self._logger.error("Delete by document failed", error=str(e))
            raise VectorStoreError(
                f"Delete by document failed: {e}",
                store="qdrant",
                operation="delete_by_document",
            ) from e

    async def count(self) -> int:
        """Get total vector count."""
        client = self._get_client()
        try:
            result = await client.count(
                collection_name=self._collection_name,
                exact=True,
            )
            return result.count
        except Exception:
            return 0

    async def health_check(self) -> bool:
        """Check if Qdrant is healthy."""
        try:
            client = self._get_client()
            await client.get_collections()
            return True
        except Exception as e:
            self._logger.warning("Qdrant health check failed", error=str(e))
            return False

    async def close(self) -> None:
        """Close the client connection."""
        if self._client:
            await self._client.close()
            self._client = None


class InMemoryVectorStore(BaseVectorStore):
    """In-memory vector store for testing and development."""

    def __init__(self, dimension: int = 512):
        self._dimension = dimension
        self._store: dict[str, dict[str, Any]] = {}
        self._logger = get_logger("cognita.vectorstore.memory")
        self._lock = asyncio.Lock()

    async def create_collection(self, dimension: int) -> None:
        self._dimension = dimension
        self._logger.info("In-memory collection created", dimension=dimension)

    async def add(self, chunks: list[Chunk]) -> int:
        async with self._lock:
            count = 0
            for chunk in chunks:
                if chunk.embedding is not None:
                    self._store[chunk.id] = {
                        "chunk": chunk,
                        "embedding": chunk.embedding,
                    }
                    count += 1
            vectorstore_collection_size.set(len(self._store))
            return count

    async def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter_conditions: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        import numpy as np

        async with self._lock:
            if not self._store:
                return []

            query_vec = np.array(query_embedding)
            results: list[tuple[float, dict[str, Any]]] = []

            for point_id, point_data in self._store.items():
                chunk = point_data["chunk"]

                # Apply filter conditions
                if filter_conditions:
                    match = all(
                        chunk.metadata.get(k) == v or k == "document_id" and chunk.document_id == v
                        for k, v in filter_conditions.items()
                    )
                    if not match:
                        continue

                emb_vec = np.array(point_data["embedding"])
                # Cosine similarity (vectors are already normalized)
                score = float(np.dot(query_vec, emb_vec))

                if score >= score_threshold:
                    results.append((score, point_data))

            # Sort by score descending
            results.sort(key=lambda x: x[0], reverse=True)
            results = results[:top_k]

            return [
                SearchResult(
                    chunk=data["chunk"],
                    score=score,
                    source_title=data["chunk"].metadata.get("title", ""),
                    source_path=data["chunk"].metadata.get("source", ""),
                )
                for score, data in results
            ]

    async def delete(self, chunk_ids: list[str]) -> int:
        async with self._lock:
            deleted = 0
            for chunk_id in chunk_ids:
                if chunk_id in self._store:
                    del self._store[chunk_id]
                    deleted += 1
            vectorstore_collection_size.set(len(self._store))
            return deleted

    async def delete_by_document(self, document_id: str) -> int:
        async with self._lock:
            to_delete = [
                pid
                for pid, data in self._store.items()
                if data["chunk"].document_id == document_id
            ]
            for pid in to_delete:
                del self._store[pid]
            vectorstore_collection_size.set(len(self._store))
            return len(to_delete)

    async def count(self) -> int:
        return len(self._store)

    async def health_check(self) -> bool:
        return True


# Factory
_vectorstore_instance: BaseVectorStore | None = None


def get_vectorstore() -> BaseVectorStore:
    """Get the singleton vector store instance based on configuration."""
    global _vectorstore_instance
    if _vectorstore_instance is None:
        settings = get_settings()
        if settings.vector_store_type == "qdrant":
            _vectorstore_instance = QdrantVectorStore(settings)
        else:
            _vectorstore_instance = InMemoryVectorStore(settings.embedding_dimension)
    return _vectorstore_instance


def set_vectorstore(store: BaseVectorStore) -> None:
    """Set a custom vector store instance (useful for testing)."""
    global _vectorstore_instance
    _vectorstore_instance = store


async def reset_vectorstore() -> None:
    """Reset the vector store instance."""
    global _vectorstore_instance
    if _vectorstore_instance is not None and hasattr(_vectorstore_instance, "close"):
        await _vectorstore_instance.close()
    _vectorstore_instance = None
