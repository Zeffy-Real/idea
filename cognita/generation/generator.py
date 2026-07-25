"""Answer generation for the RAG pipeline.

The :class:`RAGGenerator` orchestrates prompt construction, LLM invocation,
citation extraction, and response packaging. It supports synchronous
generation, token streaming, and a lightweight non-RAG ``generate_simple``
helper for utility tasks such as query expansion or summarization.
"""

from __future__ import annotations

import re
import time
from collections.abc import AsyncIterator

from cognita.config import Settings, get_settings
from cognita.core.exceptions import GenerationError
from cognita.core.llm import BaseLLM, get_llm
from cognita.core.models import (
    Citation,
    GenerationResponse,
    Message,
    SearchResult,
)
from cognita.generation.prompts import PromptBuilder
from cognita.observability.logging import get_logger
from cognita.observability.metrics import record_error

logger = get_logger("cognita.generation.generator")

# Matches inline citation markers such as [1], [2], [1][3].
_CITATION_RE = re.compile(r"\[(\d+)\]")


class RAGGenerator:
    """Generates grounded answers with citations from retrieved context.

    Dependencies (LLM, prompt builder, settings) are injected with sensible
    factory-function defaults, making the generator straightforward to test
    and to swap implementations.
    """

    def __init__(
        self,
        llm: BaseLLM | None = None,
        prompt_builder: PromptBuilder | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the RAG generator.

        Args:
            llm: LLM client implementing :class:`BaseLLM`. Defaults to the
                shared singleton returned by :func:`get_llm`.
            prompt_builder: Prompt builder instance. Defaults to a
                :class:`PromptBuilder` configured with the conversation memory
                window from settings.
            settings: Application settings. Defaults to the cached settings
                returned by :func:`get_settings`.
        """
        self._settings = settings or get_settings()
        self._llm = llm or get_llm()
        self._prompt_builder = prompt_builder or PromptBuilder(
            max_history_turns=self._settings.conversation_memory_turns
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_thinking(self, thinking: bool | None) -> bool:
        """Resolve the effective thinking mode.

        An explicit argument takes precedence; otherwise the configured default
        from settings is used.
        """
        if thinking is not None:
            return thinking
        return self._settings.enable_thinking

    def _get_model_name(self, thinking: bool) -> str:
        """Return the model name used for the current thinking mode.

        Mirrors the model selection logic of :class:`DeepSeekLLM` so the
        reported model in the response matches what the LLM actually invoked.
        """
        if thinking:
            return self._settings.deepseek_reasoning_model
        return self._settings.deepseek_chat_model

    @staticmethod
    def _sanitize_answer(content: str) -> str:
        """Normalize the raw LLM output into a non-empty answer string."""
        if not content or not content.strip():
            return (
                "I apologize, but I was unable to generate an answer based on "
                "the provided context. Please try rephrasing your question or "
                "providing additional context."
            )
        return content.strip()

    def _validate_citations(self, answer: str, num_sources: int) -> None:
        """Validate that citation markers reference existing context blocks.

        Out-of-range citations are logged as warnings but the answer is left
        unmodified to avoid corrupting the model's output.
        """
        if not answer or num_sources <= 0:
            return
        referenced = {int(n) for n in _CITATION_RE.findall(answer)}
        out_of_range = sorted(r for r in referenced if r < 1 or r > num_sources)
        if out_of_range:
            logger.warning(
                "Answer contains out-of-range citation references",
                citations=out_of_range,
                num_sources=num_sources,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def generate(
        self,
        query: str,
        search_results: list[SearchResult],
        conversation_history: list[Message] | None = None,
        thinking: bool | None = None,
    ) -> GenerationResponse:
        """Generate a grounded answer with citations for the given query.

        Args:
            query: The user's question.
            search_results: Retrieved context used to ground the answer.
            conversation_history: Optional prior conversation turns.
            thinking: Optional override for thinking mode. When ``None``, the
                configured default (:attr:`Settings.enable_thinking`) is used.

        Returns:
            A :class:`GenerationResponse` containing the answer, citations,
            usage stats, model name, optional thinking trace, and latency.

        Raises:
            GenerationError: If generation fails for any reason.
        """
        try:
            thinking_mode = self._resolve_thinking(thinking)
            messages = self._prompt_builder.build_rag_prompt(
                query=query,
                search_results=search_results,
                conversation_history=conversation_history,
            )

            logger.info(
                "Generation request",
                query=query,
                num_results=len(search_results),
                thinking=thinking_mode,
                message_count=len(messages),
            )

            start = time.perf_counter()
            content, usage, thinking_content = await self._llm.chat(
                messages,
                temperature=self._settings.temperature,
                max_tokens=self._settings.max_tokens,
                thinking=thinking_mode,
            )
            latency_ms = (time.perf_counter() - start) * 1000

            answer = self._sanitize_answer(content)
            self._validate_citations(answer, len(search_results))

            citations = [Citation.from_search_result(r) for r in search_results]
            model_name = self._get_model_name(thinking_mode)

            response = GenerationResponse(
                answer=answer,
                citations=citations,
                usage=usage,
                model=model_name,
                thinking=thinking_content,
                latency_ms=latency_ms,
            )

            logger.info(
                "Generation completed",
                model=model_name,
                latency_ms=round(latency_ms, 2),
                tokens=usage,
                citations=len(citations),
                thinking_enabled=thinking_mode,
            )
            return response

        except Exception as exc:
            logger.error(
                "Generation failed",
                query=query,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            record_error("generation_error")
            raise GenerationError(f"Answer generation failed: {exc}") from exc

    async def generate_stream(
        self,
        query: str,
        search_results: list[SearchResult],
        conversation_history: list[Message] | None = None,
        thinking: bool | None = None,
    ) -> AsyncIterator[str]:
        """Stream the generated answer token-by-token from the LLM.

        Args:
            query: The user's question.
            search_results: Retrieved context used to ground the answer.
            conversation_history: Optional prior conversation turns.
            thinking: Optional override for thinking mode.

        Yields:
            String chunks of the generated answer. On failure, a final error
            message string is yielded instead of raising.
        """
        try:
            thinking_mode = self._resolve_thinking(thinking)
            messages = self._prompt_builder.build_rag_prompt(
                query=query,
                search_results=search_results,
                conversation_history=conversation_history,
            )

            logger.info(
                "Stream generation request",
                query=query,
                num_results=len(search_results),
                thinking=thinking_mode,
                message_count=len(messages),
            )

            async for chunk in self._llm.chat_stream(
                messages,
                temperature=self._settings.temperature,
                max_tokens=self._settings.max_tokens,
                thinking=thinking_mode,
            ):
                yield chunk

            logger.info(
                "Stream generation completed",
                thinking_enabled=thinking_mode,
            )

        except Exception as exc:
            logger.error(
                "Stream generation failed",
                query=query,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            record_error("generation_error")
            yield f"\n\n[Generation error: {exc}]"

    async def generate_simple(self, query: str, context: str) -> str:
        """Generate a plain answer from a query and raw context (no RAG pipeline).

        Intended for utility tasks such as query expansion, summarization, or
        standalone QA that do not require citation extraction.

        Args:
            query: The user's question or instruction.
            context: Raw context text to condition the answer.

        Returns:
            The generated answer string.

        Raises:
            GenerationError: If generation fails for any reason.
        """
        try:
            messages: list[Message] = [
                Message(
                    role="system",
                    content=(
                        "You are a helpful assistant. Answer the user's request "
                        "based on the provided context. Be concise, accurate, and "
                        "respond in the same language as the request."
                    ),
                ),
                Message(
                    role="user",
                    content=f"Context:\n{context}\n\nQuestion:\n{query}",
                ),
            ]

            logger.debug(
                "Simple generation request",
                query=query,
                context_length=len(context),
            )

            start = time.perf_counter()
            content, _usage, _thinking = await self._llm.chat(
                messages,
                temperature=self._settings.temperature,
                max_tokens=self._settings.max_tokens,
                thinking=False,
            )
            latency_ms = (time.perf_counter() - start) * 1000

            logger.info(
                "Simple generation completed",
                latency_ms=round(latency_ms, 2),
            )
            return self._sanitize_answer(content)

        except Exception as exc:
            logger.error(
                "Simple generation failed",
                query=query,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            record_error("generation_error")
            raise GenerationError(f"Simple generation failed: {exc}") from exc
