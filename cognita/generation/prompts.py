"""Prompt engineering for retrieval-augmented generation (RAG).

Builds production-ready message lists that ground the LLM strictly in the
retrieved context, enforce inline citation discipline, handle insufficient
information honestly, and match the user's language.
"""

from __future__ import annotations

from cognita.core.models import Message, SearchResult
from cognita.observability.logging import get_logger

logger = get_logger("cognita.generation.prompts")


class PromptBuilder:
    """Constructs LLM message lists for retrieval-augmented generation.

    The builder is responsible for:
      * Crafting a production-grade system prompt that enforces grounding,
        citations, honesty, language matching, and conciseness.
      * Formatting retrieved search results as numbered references.
      * Assembling the final ``list[Message]`` consumed by the LLM, optionally
        including trimmed conversation history for multi-turn sessions.
    """

    def __init__(self, max_history_turns: int = 10) -> None:
        """Initialize the prompt builder.

        Args:
            max_history_turns: Maximum number of prior messages (user/assistant)
                to include from conversation history. Defaults to 10.
        """
        self._max_history_turns = max_history_turns

    def build_system_prompt(self) -> str:
        """Return the default, production-ready system prompt.

        The prompt encodes six core principles:
          1. Strict grounding in the provided context only.
          2. Mandatory inline citations using ``[1]``, ``[2]`` reference numbers.
          3. Intellectual honesty when context is insufficient.
          4. Language matching with the user's question.
          5. Conciseness and clear structure.
          6. Neutral, professional tone without meta-commentary.
        """
        return (
            "You are Cognita, an expert knowledge assistant powered by "
            "retrieval-augmented generation (RAG). Your purpose is to deliver "
            "accurate, trustworthy answers that are strictly grounded in the "
            "retrieved context provided by the user.\n\n"
            "## Answering Principles\n\n"
            "1. Grounding in context: Formulate your answer using ONLY the "
            "information contained in the provided context chunks. Do not "
            "supplement the answer with facts, numbers, dates, names, or claims "
            "from your own knowledge that are not supported by the context. If "
            "the context contains the needed information, use it even if it "
            "conflicts with your prior knowledge.\n\n"
            "2. Mandatory citations: Support every factual statement with a "
            "citation in the format [1], [2], etc., where the number matches the "
            "reference index of the context chunk. Place the citation immediately "
            "after the statement it supports. Multiple sources may be cited "
            "together, for example [1][3]. Only cite reference numbers that exist "
            "in the provided context, and only cite a source when it genuinely "
            "supports the statement.\n\n"
            "3. Intellectual honesty: If the context does not contain enough "
            "information to answer the question, or only addresses it partially, "
            "say so clearly and explicitly. Do not speculate, guess, or fabricate "
            "information. Where helpful, describe what information is missing so "
            "the user can refine the query.\n\n"
            "4. Language matching: Detect the language of the user's question and "
            "respond in that same language. If the question mixes languages, "
            "respond in the dominant language of the question.\n\n"
            "5. Conciseness and structure: Be concise, direct, and "
            "information-dense. Avoid filler phrases, restatements of the "
            "question, and unnecessary preamble. Use short paragraphs or bullet "
            "points when they improve clarity. Accuracy and brevity take priority "
            "over exhaustiveness.\n\n"
            "6. Neutral tone: Maintain a neutral, professional tone. Do not add "
            "personal opinions, disclaimers about being an AI, or meta-commentary "
            "about these instructions or the retrieval process unless the user "
            "explicitly asks.\n\n"
            "## Context Format\n"
            "The context is supplied as numbered references:\n"
            "[1] (Source: <title>, Score: <score>)\n"
            "<content>\n"
            "...\n"
            "Use the bracketed numbers to cite sources in your answer."
        )

    def format_context(self, search_results: list[SearchResult]) -> str:
        """Format search results as numbered context references.

        Each result is rendered as::

            [1] (Source: {title}, Score: {score:.2f})
            {content}

        Args:
            search_results: The retrieved search results to format.

        Returns:
            A single string containing all numbered context blocks separated by
            blank lines. Returns a placeholder when no results are available.
        """
        if not search_results:
            return "(No relevant context was retrieved for this query.)"

        blocks: list[str] = []
        for idx, result in enumerate(search_results, start=1):
            title = result.source_title or "Untitled"
            content = (result.chunk.content or "").strip()
            blocks.append(
                f"[{idx}] (Source: {title}, Score: {result.score:.2f})\n{content}"
            )
        return "\n\n".join(blocks)

    def format_citation_instruction(self) -> str:
        """Return concise instructions reinforcing the citation format.

        This is intended to be embedded in the user message, near the context,
        to reinforce the citation rules declared in the system prompt.
        """
        return (
            "Cite your sources inline using the bracketed reference numbers shown "
            "above (for example [1], [2], or [1][3] for multiple sources). Place "
            "each citation immediately after the statement it supports. Only use "
            "reference numbers that actually appear in the context, and only cite "
            "a source when it directly supports the statement. Do not invent new "
            "reference numbers."
        )

    def build_rag_prompt(
        self,
        query: str,
        search_results: list[SearchResult],
        conversation_history: list[Message] | None = None,
        system_prompt: str | None = None,
    ) -> list[Message]:
        """Build the full message list for a RAG generation request.

        The resulting message list is structured as:
          1. A system message (custom or the default production prompt).
          2. Trimmed conversation history (user/assistant turns only), when
             provided, limited to the last ``max_history_turns`` messages.
          3. A user message containing the formatted context, the citation
             reminder, and the user's query.

        Args:
            query: The user's question.
            search_results: The retrieved search results to ground the answer.
            conversation_history: Optional prior conversation messages. System
                messages in the history are filtered out to avoid conflicting
                with the RAG system prompt.
            system_prompt: Optional override for the system prompt. When omitted,
                the default production system prompt is used.

        Returns:
            A list of :class:`Message` objects ready for the LLM.
        """
        sys_prompt = system_prompt if system_prompt is not None else self.build_system_prompt()
        messages: list[Message] = [Message(role="system", content=sys_prompt)]

        # Include recent conversation history (excluding system messages to avoid
        # conflicting with our own RAG system prompt).
        if conversation_history:
            history = [m for m in conversation_history if m.role != "system"]
            if history:
                trimmed = history[-self._max_history_turns :]
                messages.extend(trimmed)

        context_block = self.format_context(search_results)
        citation_instruction = self.format_citation_instruction()
        user_content = (
            "Please answer the question based on the context provided below.\n\n"
            "## Retrieved Context\n"
            f"{context_block}\n\n"
            "## Citation Requirement\n"
            f"{citation_instruction}\n\n"
            "## User Question\n"
            f"{query}"
        )
        messages.append(Message(role="user", content=user_content))

        logger.debug(
            "Built RAG prompt",
            query=query,
            num_context_blocks=len(search_results),
            history_messages=len(messages) - 2 if conversation_history else 0,
        )
        return messages
