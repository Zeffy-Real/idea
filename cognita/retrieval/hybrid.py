"""
Hybrid retrieval engine.

Combines query embedding with vector-store similarity search. The retriever
purposely over-fetches candidates (2x the requested ``top_k``) so that a
downstream cross-encoder reranker has enough headroom to reorder results.

The component is designed for production:
  * dependency-injectable embedding/vectorstore (with factory defaults),
  * structured logging,
  * Prometheus metrics for every request, and
  * strict error containment -- all failures surface as ``RetrievalError``.
"""

from __future__ import annotations

import time
from typing import Any

from cognita.config import Settings, get_settings
from cognita.core.embedding import LocalEmbedding, get_embedding
from cognita.core.exceptions import RetrievalError
from cognita.core.models import Message, SearchResult
from cognita.core.vectorstore import BaseVectorStore, get_vectorstore
from cognita.observability.logging import get_logger
from cognita.observability.metrics import (
    retrieval_duration,
    retrieval_requests_total,
    retrieval_results_count,
    retrieval_score,
)

logger = get_logger("cognita.retrieval.hybrid")


class HybridRetriever:
    """Embeds a query and searches the vector store for relevant chunks.

    The retriever is intentionally decoupled from the reranker: it returns an
    over-sampled candidate set so callers (or a pipeline orchestrator) can
    optionally apply :class:`CrossEncoderReranker` afterwards.
    """

    #: Number of trailing conversation turns folded into a contextual query.
    _context_window: int = 4

    def __init__(
        self,
        embedding: LocalEmbedding | None = None,
        vectorstore: BaseVectorStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._embedding = embedding or get_embedding()
        self._vectorstore = vectorstore or get_vectorstore()
        self._logger = get_logger("cognita.retrieval.hybrid")

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        filter_conditions: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Retrieve relevant chunks for a single query.

        Args:
            query: The user query text.
            top_k: Number of results desired. Falls back to
                ``settings.retrieval_top_k`` when ``None``. The vector store is
                queried for ``top_k * 2`` candidates to support reranking.
            score_threshold: Minimum similarity score. Falls back to
                ``settings.retrieval_score_threshold`` when ``None``.
            filter_conditions: Optional metadata filters forwarded to the
                vector store.

        Returns:
            A list of :class:`SearchResult` sorted by descending score.

        Raises:
            RetrievalError: If the query is empty or any underlying step
                (embedding / search) fails.
        """
        start = time.perf_counter()
        status = "success"

        try:
            if not query or not query.strip():
                raise RetrievalError("Query cannot be empty")

            effective_top_k = (
                top_k if top_k is not None else self._settings.retrieval_top_k
            )
            effective_threshold = (
                score_threshold
                if score_threshold is not None
                else self._settings.retrieval_score_threshold
            )

            self._logger.debug(
                "Retrieval started",
                query_length=len(query),
                top_k=effective_top_k,
                score_threshold=effective_threshold,
                has_filter=bool(filter_conditions),
            )

            # 1. Embed the query.
            query_embedding = await self._embedding.embed_async(query)

            # 2. Over-fetch candidates (2x) to give a reranker room to work.
            fetch_k = effective_top_k * 2
            results = await self._vectorstore.search(
                query_embedding,
                top_k=fetch_k,
                score_threshold=effective_threshold,
                filter_conditions=filter_conditions,
            )

            # 3. Record per-result score metrics.
            for result in results:
                retrieval_score.observe(result.score)

            elapsed = time.perf_counter() - start
            retrieval_duration.observe(elapsed)
            retrieval_results_count.observe(len(results))

            self._logger.info(
                "Retrieval completed",
                results=len(results),
                top_k=effective_top_k,
                latency_ms=round(elapsed * 1000, 2),
            )

            return results

        except RetrievalError:
            status = "error"
            raise
        except Exception as exc:
            status = "error"
            self._logger.error(
                "Retrieval failed",
                error=str(exc),
                query_length=len(query) if query else 0,
            )
            raise RetrievalError(f"Retrieval failed: {exc}") from exc
        finally:
            retrieval_requests_total.labels(status=status).inc()

    async def retrieve_with_context(
        self,
        query: str,
        conversation_history: list[Message] | None = None,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Retrieve results using conversation context to refine the query.

        When ``conversation_history`` is provided, the most recent turns are
        folded into a contextual query string that helps disambiguate pronouns
        and follow-up questions. When no history is given, this is equivalent
        to :meth:`retrieve`.

        Args:
            query: The current user query.
            conversation_history: Prior conversation turns (newest last).
            **kwargs: Forwarded to :meth:`retrieve`
                (``top_k``, ``score_threshold``, ``filter_conditions``).

        Returns:
            A list of :class:`SearchResult`.
        """
        if not conversation_history:
            return await self.retrieve(query, **kwargs)

        contextual_query = self._build_contextual_query(query, conversation_history)

        self._logger.debug(
            "Retrieval with context",
            history_turns=len(conversation_history),
            original_query_length=len(query),
            contextual_query_length=len(contextual_query),
        )

        return await self.retrieve(contextual_query, **kwargs)

    def _build_contextual_query(
        self, query: str, conversation_history: list[Message]
    ) -> str:
        """Combine recent conversation turns with the current query.

        Only ``user`` and ``assistant`` messages with non-empty content are
        used. The original query is always preserved verbatim as the final
        "Current question" so the embedding model prioritises it.
        """
        recent = conversation_history[-self._context_window :]

        context_parts: list[str] = []
        for msg in recent:
            if msg.role in ("user", "assistant") and msg.content and msg.content.strip():
                context_parts.append(f"{msg.role.capitalize()}: {msg.content.strip()}")

        if not context_parts:
            return query

        context = "\n".join(context_parts)
        return (
            f"Previous conversation context:\n{context}\n\n"
            f"Current question: {query}"
        )
