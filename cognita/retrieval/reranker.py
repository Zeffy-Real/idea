"""
Cross-encoder reranker.

Reorders retrieval results using a cross-encoder model
(e.g. ``BAAI/bge-reranker-base``) that jointly encodes the (query, document)
pair. Cross-encoders are far more accurate than bi-encoder cosine similarity
but also far more expensive, so they are applied only to a small candidate set
produced by the vector store.

Design notes:
  * The model is lazily loaded on first use and guarded by a ``threading.Lock``
    so concurrent coroutines cannot trigger duplicate loads.
  * ``sentence-transformers`` is an optional dependency: if it is missing or the
    model cannot be loaded, every rerank call degrades gracefully to returning
    the input results unchanged (truncated to ``top_k``).
  * Model inference runs in a thread pool via ``run_in_executor`` so the
    event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

from cognita.config import Settings, get_settings
from cognita.core.models import SearchResult
from cognita.observability.logging import get_logger

logger = get_logger("cognita.retrieval.reranker")


class CrossEncoderReranker:
    """Rerank retrieval results with a sentence-transformers cross-encoder."""

    #: Class-level lock serialising the one-time model load.
    _load_lock = threading.Lock()

    def __init__(
        self,
        model_name: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._model_name = model_name or self._settings.rerank_model
        self._model: Any = None
        # Tri-state: None = not attempted, True = loaded, False = unavailable.
        self._available: bool | None = None
        self._logger = get_logger("cognita.retrieval.reranker")

    # ------------------------------------------------------------------ #
    # Model lifecycle
    # ------------------------------------------------------------------ #
    def _load_model(self) -> Any:
        """Lazily load the cross-encoder model (thread-safe).

        Returns the loaded model instance, or ``None`` if the dependency is
        missing or loading failed. Once a load has failed, subsequent calls
        short-circuit without retrying.
        """
        if self._model is not None:
            return self._model
        if self._available is False:
            return None

        with self._load_lock:
            # Re-check inside the lock to avoid duplicate loads.
            if self._model is not None:
                return self._model
            if self._available is False:
                return None

            try:
                from sentence_transformers import CrossEncoder

                start = time.perf_counter()
                self._logger.info(
                    "Loading reranker model", model=self._model_name
                )
                self._model = CrossEncoder(self._model_name)
                elapsed = time.perf_counter() - start
                self._available = True
                self._logger.info(
                    "Reranker model loaded",
                    model=self._model_name,
                    load_time_ms=round(elapsed * 1000, 2),
                )
            except ImportError as exc:
                self._available = False
                self._model = None
                self._logger.warning(
                    "sentence-transformers not installed; reranking disabled. "
                    "Run: pip install sentence-transformers",
                    error=str(exc),
                )
            except Exception as exc:
                self._available = False
                self._model = None
                self._logger.warning(
                    "Failed to load reranker model; reranking disabled",
                    model=self._model_name,
                    error=str(exc),
                )

        return self._model

    def is_available(self) -> bool:
        """Return ``True`` if the reranker model is loaded and ready."""
        return self._model is not None

    # ------------------------------------------------------------------ #
    # Reranking
    # ------------------------------------------------------------------ #
    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Rerank ``results`` for ``query`` using the cross-encoder.

        Args:
            query: The user query.
            results: Candidate :class:`SearchResult` objects (typically the
                over-fetched output of :class:`HybridRetriever`).
            top_k: Maximum number of results to return.

        Returns:
            Reranked results sorted by cross-encoder score, truncated to
            ``top_k``. If the reranker is unavailable or inference fails, the
            original results (truncated to ``top_k``) are returned unchanged.
        """
        # Graceful degradation: nothing to rerank or reranker not ready.
        if not results:
            return results

        model = self._load_model()
        if model is None:
            self._logger.debug(
                "Reranker unavailable; returning original results",
                count=len(results),
                top_k=top_k,
            )
            return results[:top_k]

        pairs = [(query, result.chunk.content) for result in results]

        try:
            loop = asyncio.get_running_loop()
            scores = await loop.run_in_executor(None, model.predict, pairs)
        except Exception as exc:
            self._logger.warning(
                "Reranking inference failed; returning original results",
                error=str(exc),
                count=len(results),
            )
            return results[:top_k]

        # Pair each result with its new cross-encoder score and sort descending.
        scored = list(zip(results, scores))
        scored.sort(key=lambda item: float(item[1]), reverse=True)

        reranked: list[SearchResult] = []
        for result, new_score in scored[:top_k]:
            reranked.append(result.model_copy(update={"score": float(new_score)}))

        self._logger.debug(
            "Rerank completed",
            candidates=len(results),
            returned=len(reranked),
            top_score=reranked[0].score if reranked else 0.0,
        )
        return reranked

    async def rerank_batch(
        self,
        queries: list[str],
        results_per_query: list[list[SearchResult]],
        top_k: int = 5,
    ) -> list[list[SearchResult]]:
        """Rerank multiple query/result sets concurrently.

        Args:
            queries: One query per item.
            results_per_query: Candidate results aligned with ``queries``.
            top_k: Maximum results to return per query.

        Returns:
            A list of reranked result lists, aligned with the inputs.

        Raises:
            ValueError: If ``queries`` and ``results_per_query`` differ in
                length.
        """
        if len(queries) != len(results_per_query):
            raise ValueError(
                "queries and results_per_query must have the same length: "
                f"{len(queries)} != {len(results_per_query)}"
            )

        if not queries:
            return []

        tasks = [
            self.rerank(query, results, top_k=top_k)
            for query, results in zip(queries, results_per_query, strict=True)
        ]
        return await asyncio.gather(*tasks)
