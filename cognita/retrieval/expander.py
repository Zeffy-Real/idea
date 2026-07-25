"""
Query expansion strategies.

Improves retrieval recall by generating additional search queries from an LLM:

  * :meth:`QueryExpander.expand` -- produces alternative phrasings of the
    query so the embedding search can match documents that use different
    vocabulary.
  * :meth:`QueryExpander.expand_with_hyde` -- implements HyDE (Hypothetical
    Document Embeddings): the LLM drafts a hypothetical answer, which is then
    used as an additional query. The answer's embedding tends to be closer to
    the real target passages than the question's embedding.

Both methods degrade gracefully: when no LLM is configured they simply return
``[query]`` so the retrieval pipeline keeps working.
"""

from __future__ import annotations

import re

from cognita.config import Settings, get_settings
from cognita.core.llm import DeepSeekLLM
from cognita.core.models import Message
from cognita.observability.logging import get_logger

logger = get_logger("cognita.retrieval.expander")


class QueryExpander:
    """Generate query variations and hypothetical documents via an LLM."""

    def __init__(
        self,
        llm: DeepSeekLLM | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._llm = llm
        self._logger = get_logger("cognita.retrieval.expander")

    # ------------------------------------------------------------------ #
    # Query expansion
    # ------------------------------------------------------------------ #
    async def expand(self, query: str, num_variations: int = 3) -> list[str]:
        """Generate alternative phrasings of ``query``.

        Args:
            query: The original user query.
            num_variations: Number of alternative phrasings to request.

        Returns:
            A list whose first element is always the original query, followed
            by up to ``num_variations`` deduplicated alternatives. If no LLM is
            configured or generation fails, returns ``[query]``.
        """
        if not query or not query.strip():
            return []

        if self._llm is None:
            self._logger.debug(
                "No LLM configured; returning original query only",
            )
            return [query]

        system_prompt = (
            "You are a query expansion assistant for a retrieval-augmented "
            "generation system. Given a user query, generate alternative "
            "phrasings that capture the same intent from different angles "
            "(for example: synonyms, a more specific formulation, a more "
            "general formulation, and a natural-language rephrasing). "
            f"Produce exactly {num_variations} alternatives. "
            "Return ONLY the alternative queries, one per line. "
            "Do not number them, do not prefix with bullets, do not add "
            "explanations, and do not repeat the original query."
        )
        user_prompt = (
            f"Original query: {query}\n\n"
            f"Generate {num_variations} alternative phrasings:"
        )

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]

        try:
            content, _usage, _thinking = await self._llm.chat(
                messages, temperature=0.7, max_tokens=256
            )
        except Exception as exc:
            self._logger.warning(
                "Query expansion failed; returning original query only",
                error=str(exc),
            )
            return [query]

        variations = self._parse_variations(content, num_variations)

        # Always include the original query first, then dedupe alternatives.
        expanded: list[str] = [query]
        seen = {query.strip().lower()}
        for variation in variations:
            cleaned = variation.strip()
            key = cleaned.lower()
            if cleaned and key not in seen:
                expanded.append(cleaned)
                seen.add(key)

        self._logger.debug(
            "Query expansion completed",
            original=query,
            variations_generated=len(expanded) - 1,
        )
        return expanded

    # ------------------------------------------------------------------ #
    # HyDE
    # ------------------------------------------------------------------ #
    async def expand_with_hyde(self, query: str) -> list[str]:
        """Generate a hypothetical document for ``query`` (HyDE).

        The LLM is asked to write a short, factual answer to the query. That
        hypothetical document is returned alongside the original query so the
        caller can embed and search with both.

        Args:
            query: The original user query.

        Returns:
            ``[original_query, hypothetical_document]``. If no LLM is
            configured, generation fails, or the document is empty, returns
            ``[query]``.
        """
        if not query or not query.strip():
            return []

        if self._llm is None:
            self._logger.debug(
                "No LLM configured; HyDE disabled, returning original query only",
            )
            return [query]

        system_prompt = (
            "You are a knowledgeable assistant. Given a question, write a "
            "short, factual hypothetical document (2-4 sentences) that would "
            "directly answer the question. Write only the document content -- "
            "no preamble, no meta commentary, no headings. This document will "
            "be used to retrieve relevant passages from a knowledge base."
        )
        user_prompt = f"Question: {query}\n\nHypothetical answer document:"

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]

        try:
            content, _usage, _thinking = await self._llm.chat(
                messages, temperature=0.7, max_tokens=256
            )
        except Exception as exc:
            self._logger.warning(
                "HyDE generation failed; returning original query only",
                error=str(exc),
            )
            return [query]

        hypothetical = content.strip() if content else ""
        if not hypothetical:
            self._logger.debug(
                "HyDE produced an empty document; returning original query only",
            )
            return [query]

        self._logger.debug(
            "HyDE generation completed",
            original=query,
            hypothetical_length=len(hypothetical),
        )
        return [query, hypothetical]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_variations(content: str, num_variations: int) -> list[str]:
        """Parse raw LLM output into a clean list of variation strings.

        Strips common list markers (``1.``, ``2)``, ``-``, ``*``) and
        surrounding quote characters, drops blanks, and truncates to
        ``num_variations``.
        """
        if not content:
            return []

        cleaned: list[str] = []
        for raw_line in content.strip().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            # Remove leading list markers: "1.", "2)", "-", "*", etc.
            line = re.sub(r"^\s*(?:\d+[\.\)]|[-*•])\s*", "", line).strip()
            # Strip surrounding quotes (straight and curly).
            line = line.strip("\"'“”‘’")
            if line:
                cleaned.append(line)
        return cleaned[:num_variations]
