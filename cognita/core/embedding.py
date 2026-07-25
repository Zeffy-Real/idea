"""
Embedding abstraction layer with local sentence-transformers implementation.
Provides text embedding capabilities without external API calls.
Supports batch processing and lazy model loading.
"""

from __future__ import annotations

import asyncio
import threading
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from cognita.config import Settings, get_settings
from cognita.core.exceptions import EmbeddingError
from cognita.observability.logging import get_logger
from cognita.observability.metrics import (
    embedding_request_duration,
    embedding_requests_total,
)

logger = get_logger(__name__)


class BaseEmbedding(ABC):
    """Abstract embedding interface."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding vector dimension."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in batch."""

    @abstractmethod
    async def embed_async(self, text: str) -> list[float]:
        """Async embed a single text."""

    @abstractmethod
    async def embed_batch_async(self, texts: list[str]) -> list[list[float]]:
        """Async embed multiple texts in batch."""


class LocalEmbedding(BaseEmbedding):
    """Local embedding using sentence-transformers.

    Runs entirely on-device — no API calls, no data leaving the machine.
    Model is lazily loaded on first use and cached for subsequent calls.
    """

    _model: Any = None  # SentenceTransformer instance
    _lock = threading.Lock()
    _instance: LocalEmbedding | None = None

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._logger = get_logger("cognita.embedding.local")
        self._model_name = self._settings.embedding_model
        self._device = self._settings.embedding_device
        self._dimension = self._settings.embedding_dimension

    @classmethod
    def get_instance(cls, settings: Settings | None = None) -> LocalEmbedding:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls(settings)
        return cls._instance

    def _load_model(self) -> Any:
        """Lazily load the sentence-transformers model."""
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model

            self._logger.info("Loading embedding model", model=self._model_name, device=self._device)
            start = time.perf_counter()

            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(
                    self._model_name,
                    device=self._device,
                    cache_folder=str(self._settings.data_dir / "models"),
                )
                # Verify dimension matches config
                actual_dim = self._model.get_sentence_embedding_dimension()
                if actual_dim != self._dimension:
                    self._logger.warning(
                        "Embedding dimension mismatch, updating from config",
                        config_dim=self._dimension,
                        actual_dim=actual_dim,
                    )
                    self._dimension = actual_dim

                elapsed = time.perf_counter() - start
                self._logger.info(
                    "Embedding model loaded",
                    model=self._model_name,
                    dimension=self._dimension,
                    load_time_ms=round(elapsed * 1000, 2),
                )

            except ImportError as e:
                raise EmbeddingError(
                    "sentence-transformers not installed. Run: pip install sentence-transformers",
                    model=self._model_name,
                ) from e
            except Exception as e:
                raise EmbeddingError(
                    f"Failed to load embedding model: {e}",
                    model=self._model_name,
                ) from e

            return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        """Embed a single text."""
        if not text or not text.strip():
            return [0.0] * self._dimension

        model = self._load_model()
        start = time.perf_counter()

        try:
            embedding = model.encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            embedding = embedding.tolist()

            elapsed = time.perf_counter() - start
            embedding_requests_total.labels(status="success").inc()
            embedding_request_duration.observe(elapsed)

            return embedding

        except Exception as e:
            embedding_requests_total.labels(status="error").inc()
            self._logger.error("Embedding failed", error=str(e), text_length=len(text))
            raise EmbeddingError(f"Embedding failed: {e}", model=self._model_name) from e

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in batch."""
        if not texts:
            return []

        # Filter empty texts
        non_empty_indices = [i for i, t in enumerate(texts) if t and t.strip()]
        if not non_empty_indices:
            return [[0.0] * self._dimension for _ in texts]

        model = self._load_model()
        start = time.perf_counter()

        try:
            non_empty_texts = [texts[i] for i in non_empty_indices]
            embeddings = model.encode(
                non_empty_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=min(32, len(non_empty_texts)),
            )

            # Build result with zeros for empty texts
            result: list[list[float]] = [[0.0] * self._dimension for _ in texts]
            for idx, emb in zip(non_empty_indices, embeddings, strict=False):
                result[idx] = emb.tolist()

            elapsed = time.perf_counter() - start
            embedding_requests_total.labels(status="success").inc()
            embedding_request_duration.observe(elapsed)

            self._logger.debug(
                "Batch embedding completed",
                count=len(texts),
                latency_ms=round(elapsed * 1000, 2),
            )

            return result

        except Exception as e:
            embedding_requests_total.labels(status="error").inc()
            self._logger.error("Batch embedding failed", error=str(e), count=len(texts))
            raise EmbeddingError(f"Batch embedding failed: {e}", model=self._model_name) from e

    async def embed_async(self, text: str) -> list[float]:
        """Async embed a single text (offloads to thread pool)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed, text)

    async def embed_batch_async(self, texts: list[str]) -> list[list[float]]:
        """Async embed multiple texts (offloads to thread pool)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_batch, texts)

    def similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts."""
        emb_a = np.array(self.embed(text_a))
        emb_b = np.array(self.embed(text_b))

        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(emb_a, emb_b) / (norm_a * norm_b))


def get_embedding() -> LocalEmbedding:
    """Get the singleton embedding instance."""
    return LocalEmbedding.get_instance()


def reset_embedding() -> None:
    """Reset the embedding instance (useful for testing)."""
    LocalEmbedding._instance = None
    LocalEmbedding._model = None
