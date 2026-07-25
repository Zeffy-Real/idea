"""Unit tests for PromptBuilder."""

from __future__ import annotations

import pytest

from cognita.core.models import Chunk, Message, SearchResult
from cognita.generation.prompts import PromptBuilder


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_search_results(count: int = 3) -> list[SearchResult]:
    """Create a list of SearchResult objects for testing."""
    results: list[SearchResult] = []
    for i in range(count):
        chunk = Chunk(
            id=f"chunk-{i}",
            document_id=f"doc-{i}",
            content=f"Content of chunk {i}. This is relevant information.",
            index=i,
            token_count=10,
            metadata={"title": f"Document {i}", "source": f"/path/to/doc{i}.txt"},
        )
        results.append(
            SearchResult(
                chunk=chunk,
                score=0.9 - i * 0.1,
                source_title=f"Document {i}",
                source_path=f"/path/to/doc{i}.txt",
            )
        )
    return results


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class TestBuildSystemPrompt:
    """build_system_prompt behavior."""

    def test_returns_non_empty_string(self) -> None:
        """build_system_prompt returns a non-empty string."""
        builder = PromptBuilder()
        prompt = builder.build_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_contains_key_instructions(self) -> None:
        """The system prompt contains key instruction keywords."""
        builder = PromptBuilder()
        prompt = builder.build_system_prompt()
        prompt_lower = prompt.lower()
        # Check for core principles
        assert "cognita" in prompt_lower
        assert "context" in prompt_lower
        assert "citation" in prompt_lower
        assert "grounding" in prompt_lower or "grounded" in prompt_lower
        assert "honest" in prompt_lower or "honesty" in prompt_lower

    def test_contains_citation_format(self) -> None:
        """The system prompt mentions the [1], [2] citation format."""
        builder = PromptBuilder()
        prompt = builder.build_system_prompt()
        assert "[1]" in prompt
        assert "[2]" in prompt


class TestFormatContext:
    """format_context behavior."""

    def test_with_results_returns_numbered_context(self) -> None:
        """format_context returns numbered context blocks for results."""
        builder = PromptBuilder()
        results = _make_search_results(3)
        context = builder.format_context(results)
        assert "[1]" in context
        assert "[2]" in context
        assert "[3]" in context
        assert "Document 0" in context
        assert "Document 1" in context
        assert "Document 2" in context
        assert "Content of chunk 0" in context

    def test_with_empty_results_returns_placeholder(self) -> None:
        """format_context returns a placeholder for empty results."""
        builder = PromptBuilder()
        context = builder.format_context([])
        assert isinstance(context, str)
        assert len(context) > 0
        assert "no relevant context" in context.lower()

    def test_context_includes_score(self) -> None:
        """format_context includes the score in each block."""
        builder = PromptBuilder()
        results = _make_search_results(1)
        context = builder.format_context(results)
        assert "Score:" in context

    def test_context_blocks_separated_by_blank_lines(self) -> None:
        """Context blocks are separated by double newlines."""
        builder = PromptBuilder()
        results = _make_search_results(3)
        context = builder.format_context(results)
        blocks = context.split("\n\n")
        # At least 3 content blocks
        assert len(blocks) >= 3


class TestBuildRagPrompt:
    """build_rag_prompt behavior."""

    def test_returns_list_of_messages(self) -> None:
        """build_rag_prompt returns a list of Message objects."""
        builder = PromptBuilder()
        results = _make_search_results(2)
        messages = builder.build_rag_prompt(query="What is AI?", search_results=results)
        assert isinstance(messages, list)
        assert len(messages) >= 2
        assert all(isinstance(m, Message) for m in messages)

    def test_first_message_is_system(self) -> None:
        """The first message has role 'system'."""
        builder = PromptBuilder()
        results = _make_search_results(2)
        messages = builder.build_rag_prompt(query="What is AI?", search_results=results)
        assert messages[0].role == "system"
        assert len(messages[0].content) > 0

    def test_last_message_is_user_with_query(self) -> None:
        """The last message has role 'user' and contains the query."""
        builder = PromptBuilder()
        results = _make_search_results(2)
        query = "What is deep learning?"
        messages = builder.build_rag_prompt(query=query, search_results=results)
        last_msg = messages[-1]
        assert last_msg.role == "user"
        assert query in last_msg.content

    def test_user_message_contains_context(self) -> None:
        """The user message contains the formatted context."""
        builder = PromptBuilder()
        results = _make_search_results(2)
        messages = builder.build_rag_prompt(query="Q", search_results=results)
        user_content = messages[-1].content
        assert "[1]" in user_content
        assert "[2]" in user_content

    def test_citation_instruction_is_present(self) -> None:
        """The citation instruction is present in the user message."""
        builder = PromptBuilder()
        results = _make_search_results(2)
        messages = builder.build_rag_prompt(query="Q", search_results=results)
        user_content = messages[-1].content
        assert "citation" in user_content.lower()
        assert "[1]" in user_content

    def test_with_conversation_history_includes_history(self) -> None:
        """build_rag_prompt includes conversation history messages."""
        builder = PromptBuilder(max_history_turns=10)
        results = _make_search_results(1)
        history = [
            Message(role="user", content="Previous question"),
            Message(role="assistant", content="Previous answer"),
        ]
        messages = builder.build_rag_prompt(
            query="New question",
            search_results=results,
            conversation_history=history,
        )
        # system + 2 history + user = 4
        assert len(messages) == 4
        assert messages[1].content == "Previous question"
        assert messages[2].content == "Previous answer"
        assert messages[3].role == "user"

    def test_system_messages_in_history_are_filtered(self) -> None:
        """System messages in conversation history are excluded."""
        builder = PromptBuilder(max_history_turns=10)
        results = _make_search_results(1)
        history = [
            Message(role="system", content="Old system prompt"),
            Message(role="user", content="Q1"),
            Message(role="assistant", content="A1"),
        ]
        messages = builder.build_rag_prompt(
            query="Q2",
            search_results=results,
            conversation_history=history,
        )
        # system (new) + user + assistant + user = 4
        assert len(messages) == 4
        for msg in messages[1:]:
            assert msg.role != "system"

    def test_history_trimmed_to_max_turns(self) -> None:
        """History is trimmed to max_history_turns."""
        builder = PromptBuilder(max_history_turns=2)
        results = _make_search_results(1)
        # Build a long history of 20 messages (10 user + 10 assistant).
        history: list[Message] = []
        for i in range(10):
            history.append(Message(role="user", content=f"Q{i}"))
            history.append(Message(role="assistant", content=f"A{i}"))

        messages = builder.build_rag_prompt(
            query="Final Q",
            search_results=results,
            conversation_history=history,
        )
        # system + 2 trimmed history + user = 4
        assert len(messages) == 4

    def test_custom_system_prompt_override(self) -> None:
        """A custom system_prompt overrides the default."""
        builder = PromptBuilder()
        results = _make_search_results(1)
        custom = "You are a custom assistant."
        messages = builder.build_rag_prompt(
            query="Q",
            search_results=results,
            system_prompt=custom,
        )
        assert messages[0].content == custom

    def test_empty_results_still_builds_prompt(self) -> None:
        """build_rag_prompt works with empty search results."""
        builder = PromptBuilder()
        messages = builder.build_rag_prompt(query="Q", search_results=[])
        assert len(messages) == 2  # system + user
        assert "no relevant context" in messages[-1].content.lower()
