"""
Text chunking strategies for splitting documents into embeddable chunks.

The :class:`TextChunker` implements a recursive text-splitting algorithm
inspired by LangChain's ``RecursiveCharacterTextSplitter`` but operates on
**token counts** (via ``tiktoken``) rather than raw character counts. This
ensures chunks respect model context-window limits precisely.

Splitting hierarchy (coarsest -> finest):
    1. Double newlines  (paragraphs)
    2. Single newlines  (lines)
    3. Sentence boundaries  (regex: ``(?<=[.!?])\\s+``)
    4. Whitespace       (words)

After the recursive split, small pieces are greedily merged back together
up to ``chunk_size`` tokens, preserving ``chunk_overlap`` tokens of context
between consecutive chunks so that retrieval has some boundary context.
"""

from __future__ import annotations

import re
from typing import Any

from cognita.core.exceptions import ChunkingError
from cognita.core.models import Chunk, Document
from cognita.observability.logging import get_logger

logger = get_logger("cognita.ingestion.chunkers")

# Split *after* a sentence-ending punctuation mark followed by whitespace.
_SENTENCE_PATTERN = r"(?<=[.!?])\s+"


class TextChunker:
    """Split text into token-bounded chunks with configurable overlap."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        encoding_name: str = "cl100k_base",
    ) -> None:
        if chunk_size <= 0:
            raise ChunkingError("chunk_size must be a positive integer")
        if chunk_overlap < 0:
            raise ChunkingError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ChunkingError(
                "chunk_overlap must be smaller than chunk_size "
                f"(got overlap={chunk_overlap}, size={chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._encoding_name = encoding_name

        # tiktoken encoding is loaded lazily on first use.
        self._encoding: Any = None
        self._encoding_loaded = False

        self._logger = get_logger("cognita.ingestion.chunkers")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def chunk(
        self,
        text: str,
        document_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Split *text* into :class:`Chunk` objects bound to *document_id*."""
        if not text or not text.strip():
            return []

        try:
            pieces = self._split_text(text)
        except ChunkingError:
            raise
        except Exception as exc:
            raise ChunkingError(f"Failed to split text: {exc}") from exc

        base_metadata: dict[str, Any] = dict(metadata) if metadata else {}

        chunks: list[Chunk] = []
        for index, piece in enumerate(pieces):
            token_count = self._count_tokens(piece)
            chunk_metadata: dict[str, Any] = {
                **base_metadata,
                "chunk_index": index,
                "token_count": token_count,
            }
            chunks.append(
                Chunk(
                    document_id=document_id,
                    content=piece,
                    index=index,
                    token_count=token_count,
                    metadata=chunk_metadata,
                )
            )

        self._logger.info(
            "Text chunked",
            document_id=document_id,
            chunks=len(chunks),
            total_tokens=sum(c.token_count for c in chunks),
        )
        return chunks

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Convenience wrapper that chunks a :class:`Document` directly."""
        metadata: dict[str, Any] = {
            "title": document.title,
            "source": document.source,
            "file_type": document.file_type,
            "document_id": document.id,
            **document.metadata,
        }
        return self.chunk(document.content, document.id, metadata)

    # ------------------------------------------------------------------ #
    # Token counting
    # ------------------------------------------------------------------ #

    def _count_tokens(self, text: str) -> int:
        """Return the token count of *text* using tiktoken (with fallback)."""
        if not text:
            return 0
        encoding = self._get_encoding()
        if encoding is not None:
            return len(encoding.encode(text))
        # Rough fallback: ~1 token per word (only used when tiktoken is
        # unavailable, e.g. offline environments).
        return max(1, len(text.split()))

    def _get_encoding(self) -> Any:
        """Lazily load and cache the tiktoken encoding."""
        if not self._encoding_loaded:
            self._encoding_loaded = True
            try:
                import tiktoken

                self._encoding = tiktoken.get_encoding(self._encoding_name)
            except Exception as exc:
                self._logger.warning(
                    "Failed to load tiktoken encoding; falling back to "
                    "word-based token approximation",
                    encoding=self._encoding_name,
                    error=str(exc),
                )
                self._encoding = None
        return self._encoding

    # ------------------------------------------------------------------ #
    # Recursive splitting + overlap merge
    # ------------------------------------------------------------------ #

    def _split_text(self, text: str) -> list[str]:
        """Recursively split *text* and merge into token-bounded chunks."""
        text = text.strip()
        if not text:
            return []

        # Fast path: the entire text fits in a single chunk.
        if self._count_tokens(text) <= self.chunk_size:
            return [text]

        # Coarsest -> finest separators.
        separators: list[str] = ["\n\n", "\n", _SENTENCE_PATTERN, " "]
        pieces = self._recursive_split(text, separators)

        return self._merge_with_overlap(pieces)

    def _recursive_split(
        self, text: str, separators: list[str]
    ) -> list[str]:
        """Recively split *text* until every piece fits within ``chunk_size``."""
        text = text.strip()
        if not text:
            return []

        if self._count_tokens(text) <= self.chunk_size:
            return [text]

        # No separators left — force a word-level split.
        if not separators:
            return self._split_by_words(text)

        separator = separators[0]
        remaining = separators[1:]

        # Regex separators start with a lookbehind assertion.
        if separator.startswith("(?<="):
            parts = re.split(separator, text)
        else:
            parts = text.split(separator)

        result: list[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if self._count_tokens(part) <= self.chunk_size:
                result.append(part)
            else:
                result.extend(self._recursive_split(part, remaining))

        return result

    def _split_by_words(self, text: str) -> list[str]:
        """Last-resort splitter: greedily pack words up to ``chunk_size``."""
        words = text.split()
        pieces: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for word in words:
            word_tokens = self._count_tokens(word)
            if current and current_tokens + word_tokens > self.chunk_size:
                pieces.append(" ".join(current))
                current = [word]
                current_tokens = word_tokens
            else:
                current.append(word)
                current_tokens += word_tokens

        if current:
            pieces.append(" ".join(current))

        return pieces

    def _merge_with_overlap(self, pieces: list[str]) -> list[str]:
        """Greedily merge *pieces* into chunks, keeping overlap between them."""
        if not pieces:
            return []

        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        i = 0
        while i < len(pieces):
            piece = pieces[i]
            piece_tokens = self._count_tokens(piece)

            if current and current_tokens + piece_tokens > self.chunk_size:
                # Finalise the current chunk.
                chunks.append(" ".join(current))

                # Build overlap from the tail of the just-finalised chunk.
                overlap: list[str] = []
                overlap_tokens = 0
                for p in reversed(current):
                    p_tokens = self._count_tokens(p)
                    if overlap_tokens + p_tokens > self.chunk_overlap:
                        break
                    overlap.insert(0, p)
                    overlap_tokens += p_tokens

                current = overlap
                current_tokens = overlap_tokens

                # Guard against an infinite loop: if the overlap context
                # alone already exceeds the budget for this piece, drop the
                # overlap so the piece can start a fresh chunk.
                if current and current_tokens + piece_tokens > self.chunk_size:
                    current = []
                    current_tokens = 0
                # NOTE: ``i`` is intentionally not incremented here so the
                # current piece is retried against the (possibly empty)
                # overlap context.
            else:
                current.append(piece)
                current_tokens += piece_tokens
                i += 1

        if current:
            chunks.append(" ".join(current))

        return chunks
