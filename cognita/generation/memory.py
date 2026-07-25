"""Thread-safe conversation memory for multi-turn RAG sessions.

Stores conversation messages as structured records with timestamps and
optional citation metadata. All public operations are guarded by a lock so
the memory can be shared safely across concurrent async tasks or threads.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

from cognita.core.models import Citation, Message
from cognita.observability.logging import get_logger

logger = get_logger("cognita.generation.memory")


class ConversationMemory:
    """A bounded, thread-safe buffer of conversation messages.

    Messages are stored internally as plain dicts so the memory can be
    serialized to/from JSON-compatible structures. ``max_turns`` caps the
    number of retained messages: when the limit is exceeded, the oldest
    messages are evicted first.
    """

    def __init__(self, max_turns: int = 10) -> None:
        """Initialize the conversation memory.

        Args:
            max_turns: Maximum number of messages to retain. When this limit is
                exceeded, the oldest messages are removed. Defaults to 10.
        """
        self._max_turns = max_turns
        self._messages: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _trim_locked(self) -> None:
        """Evict oldest messages until the buffer fits within ``max_turns``.

        Must be called while holding ``self._lock``.
        """
        overflow = len(self._messages) - self._max_turns
        if overflow > 0:
            del self._messages[:overflow]

    @staticmethod
    def _to_message(record: dict[str, Any]) -> Message:
        """Convert an internal record dict to a :class:`Message` object."""
        return Message(
            role=record["role"],
            content=record["content"],
            metadata=dict(record.get("metadata", {})),
        )

    # ------------------------------------------------------------------
    # Mutation API
    # ------------------------------------------------------------------
    def add_message(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append a message to the conversation, trimming if necessary.

        Args:
            role: The message role (e.g. ``"user"``, ``"assistant"``).
            content: The message content.
            metadata: Optional metadata to attach to the message.
        """
        with self._lock:
            self._messages.append(
                {
                    "role": role,
                    "content": content,
                    "metadata": metadata or {},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            self._trim_locked()
        logger.debug("Message added", role=role, total_messages=len(self._messages))

    def add_user_message(self, content: str) -> None:
        """Append a user message."""
        self.add_message(role="user", content=content)

    def add_assistant_message(
        self,
        content: str,
        citations: list[Citation] | None = None,
    ) -> None:
        """Append an assistant message, storing citations in metadata.

        Args:
            content: The assistant's answer text.
            citations: Optional list of citations supporting the answer. They are
                serialized to dicts for storage portability.
        """
        metadata: dict[str, Any] = {}
        if citations:
            metadata["citations"] = [c.model_dump() for c in citations]
        self.add_message(role="assistant", content=content, metadata=metadata)

    def clear(self) -> None:
        """Remove all messages from the conversation."""
        with self._lock:
            count = len(self._messages)
            self._messages.clear()
        logger.debug("Conversation memory cleared", cleared_messages=count)

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------
    def get_messages(self) -> list[Message]:
        """Return all retained messages as :class:`Message` objects."""
        with self._lock:
            return [self._to_message(m) for m in self._messages]

    def get_recent_messages(self, count: int | None = None) -> list[Message]:
        """Return the most recent messages as :class:`Message` objects.

        Args:
            count: Number of messages to return from the end of the buffer.
                When ``None``, all retained messages are returned.

        Returns:
            A list of :class:`Message` objects in chronological order.
        """
        with self._lock:
            if count is None or count < 0:
                records = self._messages
            else:
                records = self._messages[-count:] if count else []
            return [self._to_message(m) for m in records]

    def get_summary(self) -> str:
        """Return a human-readable text summary of the conversation.

        Useful for debugging or display surfaces.
        """
        with self._lock:
            if not self._messages:
                return "Empty conversation."

            lines = [
                f"Conversation ({len(self._messages)} messages, "
                f"max_turns={self._max_turns}):"
            ]
            for record in self._messages:
                role = record["role"]
                content = record["content"]
                preview = content if len(content) <= 120 else content[:120] + "..."
                has_citations = bool(record.get("metadata", {}).get("citations"))
                suffix = " [+citations]" if has_citations else ""
                lines.append(f"[{role}] {preview}{suffix}")
            return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Serialize the conversation memory to a JSON-compatible dict."""
        with self._lock:
            return {
                "max_turns": self._max_turns,
                "messages": [dict(m) for m in self._messages],
            }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationMemory:
        """Deserialize a conversation memory from a dict produced by ``to_dict``.

        Args:
            data: A dict with ``max_turns`` and ``messages`` keys.

        Returns:
            A reconstructed :class:`ConversationMemory` instance.
        """
        max_turns = int(data.get("max_turns", 10))
        instance = cls(max_turns=max_turns)
        messages = data.get("messages", []) or []
        instance._messages = [dict(m) for m in messages]
        instance._trim_locked()
        return instance

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        with self._lock:
            return len(self._messages)

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"ConversationMemory(messages={len(self._messages)}, "
                f"max_turns={self._max_turns})"
            )
