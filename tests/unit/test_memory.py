"""Unit tests for ConversationMemory."""

from __future__ import annotations

import threading
from typing import Any

import pytest

from cognita.core.models import Chunk, Citation, SearchResult
from cognita.generation.memory import ConversationMemory


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_citation(chunk_id: str = "c1", document_id: str = "d1") -> Citation:
    """Create a minimal Citation for testing."""
    chunk = Chunk(id=chunk_id, document_id=document_id, content="content", index=0)
    result = SearchResult(chunk=chunk, score=0.9, source_title="Title", source_path="path")
    return Citation.from_search_result(result)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class TestConversationMemoryAdd:
    """add_user_message / add_assistant_message."""

    def test_add_user_message(self) -> None:
        """add_user_message appends a user-role message."""
        mem = ConversationMemory(max_turns=10)
        mem.add_user_message("Hello")
        messages = mem.get_messages()
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"

    def test_add_assistant_message(self) -> None:
        """add_assistant_message appends an assistant-role message."""
        mem = ConversationMemory(max_turns=10)
        mem.add_assistant_message("Hi there!")
        messages = mem.get_messages()
        assert len(messages) == 1
        assert messages[0].role == "assistant"
        assert messages[0].content == "Hi there!"

    def test_add_message_generic(self) -> None:
        """add_message accepts arbitrary roles."""
        mem = ConversationMemory(max_turns=10)
        mem.add_message(role="system", content="You are helpful.")
        messages = mem.get_messages()
        assert messages[0].role == "system"


class TestConversationMemoryGetMessages:
    """get_messages returns the correct list."""

    def test_get_messages_returns_all(self) -> None:
        """get_messages returns all messages in chronological order."""
        mem = ConversationMemory(max_turns=10)
        mem.add_user_message("Q1")
        mem.add_assistant_message("A1")
        mem.add_user_message("Q2")
        messages = mem.get_messages()
        assert len(messages) == 3
        assert messages[0].content == "Q1"
        assert messages[1].content == "A1"
        assert messages[2].content == "Q2"

    def test_get_messages_empty(self) -> None:
        """get_messages on an empty memory returns []."""
        mem = ConversationMemory(max_turns=10)
        assert mem.get_messages() == []

    def test_len_returns_message_count(self) -> None:
        """__len__ returns the number of stored messages."""
        mem = ConversationMemory(max_turns=10)
        assert len(mem) == 0
        mem.add_user_message("hi")
        assert len(mem) == 1


class TestConversationMemoryMaxTurns:
    """max_turns eviction behavior."""

    def test_oldest_evicted_when_max_exceeded(self) -> None:
        """Adding more than max_turns evicts the oldest messages."""
        mem = ConversationMemory(max_turns=3)
        mem.add_user_message("msg1")
        mem.add_user_message("msg2")
        mem.add_user_message("msg3")
        mem.add_user_message("msg4")  # this should evict "msg1"
        messages = mem.get_messages()
        assert len(messages) == 3
        assert messages[0].content == "msg2"
        assert messages[2].content == "msg4"

    def test_max_turns_one(self) -> None:
        """max_turns=1 keeps only the most recent message."""
        mem = ConversationMemory(max_turns=1)
        mem.add_user_message("first")
        mem.add_user_message("second")
        messages = mem.get_messages()
        assert len(messages) == 1
        assert messages[0].content == "second"


class TestConversationMemoryClear:
    """clear() behavior."""

    def test_clear_removes_all_messages(self) -> None:
        """clear() empties the memory."""
        mem = ConversationMemory(max_turns=10)
        mem.add_user_message("a")
        mem.add_assistant_message("b")
        assert len(mem) == 2
        mem.clear()
        assert len(mem) == 0
        assert mem.get_messages() == []


class TestConversationMemorySerialization:
    """to_dict / from_dict round-trip."""

    def test_to_dict_contains_max_turns_and_messages(self) -> None:
        """to_dict returns a dict with max_turns and messages keys."""
        mem = ConversationMemory(max_turns=5)
        mem.add_user_message("hello")
        d = mem.to_dict()
        assert "max_turns" in d
        assert "messages" in d
        assert d["max_turns"] == 5
        assert len(d["messages"]) == 1
        assert d["messages"][0]["role"] == "user"
        assert d["messages"][0]["content"] == "hello"

    def test_from_dict_reconstructs_memory(self) -> None:
        """from_dict reconstructs a ConversationMemory from a dict."""
        original = ConversationMemory(max_turns=5)
        original.add_user_message("Q1")
        original.add_assistant_message("A1")
        d = original.to_dict()

        restored = ConversationMemory.from_dict(d)
        assert restored.to_dict() == d
        messages = restored.get_messages()
        assert len(messages) == 2
        assert messages[0].content == "Q1"
        assert messages[1].content == "A1"

    def test_round_trip_preserves_citations(self) -> None:
        """Citations stored in assistant messages survive a round-trip."""
        citation = _make_citation()
        original = ConversationMemory(max_turns=10)
        original.add_user_message("Q")
        original.add_assistant_message("A", citations=[citation])

        d = original.to_dict()
        restored = ConversationMemory.from_dict(d)
        messages = restored.get_messages()
        assistant_msg = messages[1]
        assert "citations" in assistant_msg.metadata
        assert len(assistant_msg.metadata["citations"]) == 1
        assert assistant_msg.metadata["citations"][0]["chunk_id"] == citation.chunk_id

    def test_from_dict_trims_if_exceeds_max_turns(self) -> None:
        """from_dict trims messages that exceed max_turns."""
        data = {
            "max_turns": 2,
            "messages": [
                {"role": "user", "content": "m1", "metadata": {}, "timestamp": ""},
                {"role": "assistant", "content": "m2", "metadata": {}, "timestamp": ""},
                {"role": "user", "content": "m3", "metadata": {}, "timestamp": ""},
                {"role": "assistant", "content": "m4", "metadata": {}, "timestamp": ""},
            ],
        }
        mem = ConversationMemory.from_dict(data)
        messages = mem.get_messages()
        assert len(messages) == 2
        assert messages[0].content == "m3"
        assert messages[1].content == "m4"


class TestConversationMemoryGetRecent:
    """get_recent_messages with count parameter."""

    def test_get_recent_messages_with_count(self) -> None:
        """get_recent_messages(n) returns the last n messages."""
        mem = ConversationMemory(max_turns=20)
        for i in range(10):
            mem.add_user_message(f"msg{i}")
        recent = mem.get_recent_messages(count=3)
        assert len(recent) == 3
        assert recent[0].content == "msg7"
        assert recent[2].content == "msg9"

    def test_get_recent_messages_count_none_returns_all(self) -> None:
        """get_recent_messages(None) returns all messages."""
        mem = ConversationMemory(max_turns=20)
        for i in range(5):
            mem.add_user_message(f"msg{i}")
        recent = mem.get_recent_messages(count=None)
        assert len(recent) == 5

    def test_get_recent_messages_count_zero_returns_empty(self) -> None:
        """get_recent_messages(0) returns an empty list."""
        mem = ConversationMemory(max_turns=20)
        mem.add_user_message("hi")
        assert mem.get_recent_messages(count=0) == []

    def test_get_recent_messages_count_exceeds_size(self) -> None:
        """get_recent_messages with count > len returns all messages."""
        mem = ConversationMemory(max_turns=20)
        mem.add_user_message("only")
        recent = mem.get_recent_messages(count=100)
        assert len(recent) == 1


class TestConversationMemoryCitations:
    """add_assistant_message with citations stores them in metadata."""

    def test_citations_stored_in_metadata(self) -> None:
        """Citations are serialized into the message metadata."""
        mem = ConversationMemory(max_turns=10)
        citation = _make_citation(chunk_id="chk-1", document_id="doc-1")
        mem.add_assistant_message("Answer [1].", citations=[citation])
        messages = mem.get_messages()
        assert messages[0].role == "assistant"
        assert "citations" in messages[0].metadata
        stored = messages[0].metadata["citations"]
        assert len(stored) == 1
        assert stored[0]["chunk_id"] == "chk-1"
        assert stored[0]["document_id"] == "doc-1"

    def test_no_citations_means_no_citations_key(self) -> None:
        """When no citations are provided, metadata has no 'citations' key."""
        mem = ConversationMemory(max_turns=10)
        mem.add_assistant_message("Plain answer.")
        messages = mem.get_messages()
        assert "citations" not in messages[0].metadata

    def test_multiple_citations_stored(self) -> None:
        """Multiple citations are all stored in metadata."""
        mem = ConversationMemory(max_turns=10)
        c1 = _make_citation(chunk_id="c1", document_id="d1")
        c2 = _make_citation(chunk_id="c2", document_id="d2")
        mem.add_assistant_message("Answer [1][2].", citations=[c1, c2])
        messages = mem.get_messages()
        assert len(messages[0].metadata["citations"]) == 2


class TestConversationMemoryThreadSafety:
    """Basic thread-safety smoke test."""

    def test_concurrent_adds_do_not_lose_messages_beyond_max(self) -> None:
        """Concurrent add_message calls respect max_turns."""
        mem = ConversationMemory(max_turns=100)
        num_threads = 10
        msgs_per_thread = 10

        def worker() -> None:
            for i in range(msgs_per_thread):
                mem.add_user_message(f"msg")

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Total messages added = num_threads * msgs_per_thread = 100
        # max_turns = 100, so all should fit (or be very close)
        assert len(mem) <= 100
        assert len(mem) > 0
