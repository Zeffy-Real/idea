"""Unit tests for InMemoryVectorStore."""

from __future__ import annotations

import math
from typing import Any

import pytest

from cognita.core.models import Chunk, SearchResult
from cognita.core.vectorstore import InMemoryVectorStore


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _unit_vector(dim: int, axis: int) -> list[float]:
    """Return a unit vector along the given axis (1 at position *axis*, 0 elsewhere)."""
    vec = [0.0] * dim
    vec[axis] = 1.0
    return vec


def _scaled_unit_vector(dim: int, components: dict[int, float]) -> list[float]:
    """Return a unit vector with the given components, normalized to unit length.

    Example: _scaled_unit_vector(512, {0: 0.6, 1: 0.8}) produces a unit vector
    with 0.6 on axis 0 and 0.8 on axis 1.
    """
    vec = [0.0] * dim
    for axis, value in components.items():
        vec[axis] = value
    # Normalize
    magnitude = math.sqrt(sum(v * v for v in vec))
    if magnitude > 0:
        vec = [v / magnitude for v in vec]
    return vec


def _make_chunk(
    chunk_id: str,
    document_id: str,
    content: str,
    embedding: list[float],
    metadata: dict[str, Any] | None = None,
) -> Chunk:
    """Create a Chunk with an embedding for testing."""
    return Chunk(
        id=chunk_id,
        document_id=document_id,
        content=content,
        index=0,
        token_count=len(content.split()),
        metadata=metadata or {},
        embedding=embedding,
    )


DIM = 512


# --------------------------------------------------------------------------- #
# create_collection
# --------------------------------------------------------------------------- #


class TestCreateCollection:
    """Tests for create_collection."""

    async def test_create_collection_sets_dimension(self) -> None:
        """create_collection updates the store's dimension."""
        store = InMemoryVectorStore(dimension=256)
        await store.create_collection(512)
        assert store._dimension == 512

    async def test_create_collection_does_not_clear_store(self) -> None:
        """create_collection does not remove existing data."""
        store = InMemoryVectorStore(dimension=DIM)
        await store.create_collection(DIM)
        chunk = _make_chunk("c1", "d1", "hello", _unit_vector(DIM, 0))
        await store.add([chunk])
        assert await store.count() == 1
        await store.create_collection(DIM)
        assert await store.count() == 1


# --------------------------------------------------------------------------- #
# add
# --------------------------------------------------------------------------- #


class TestAdd:
    """Tests for add."""

    async def test_add_returns_correct_count(self) -> None:
        """add returns the number of chunks with embeddings."""
        store = InMemoryVectorStore(dimension=DIM)
        await store.create_collection(DIM)
        chunks = [
            _make_chunk("c1", "d1", "content 1", _unit_vector(DIM, 0)),
            _make_chunk("c2", "d1", "content 2", _unit_vector(DIM, 1)),
            _make_chunk("c3", "d1", "content 3", _unit_vector(DIM, 2)),
        ]
        count = await store.add(chunks)
        assert count == 3

    async def test_add_empty_list_returns_zero(self) -> None:
        """add with an empty list returns 0."""
        store = InMemoryVectorStore(dimension=DIM)
        await store.create_collection(DIM)
        assert await store.add([]) == 0

    async def test_add_chunks_without_embeddings_not_counted(self) -> None:
        """Chunks without embeddings are skipped."""
        store = InMemoryVectorStore(dimension=DIM)
        await store.create_collection(DIM)
        chunk_no_emb = Chunk(id="c1", document_id="d1", content="no embedding")
        count = await store.add([chunk_no_emb])
        assert count == 0

    async def test_add_mixed_chunks(self) -> None:
        """Only chunks with embeddings are added."""
        store = InMemoryVectorStore(dimension=DIM)
        await store.create_collection(DIM)
        chunks = [
            _make_chunk("c1", "d1", "with emb", _unit_vector(DIM, 0)),
            Chunk(id="c2", document_id="d1", content="no emb"),
        ]
        count = await store.add(chunks)
        assert count == 1
        assert await store.count() == 1


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #


class TestSearch:
    """Tests for search."""

    async def test_search_returns_results_sorted_by_score(self) -> None:
        """Search results are sorted by score in descending order."""
        store = InMemoryVectorStore(dimension=DIM)
        await store.create_collection(DIM)
        # Chunk A: aligned with axis 0 -> score 1.0 with query along axis 0
        # Chunk B: 45 degrees between axes 0 and 1 -> score ~0.707
        # Chunk C: aligned with axis 1 -> score 0.0 with query along axis 0
        chunks = [
            _make_chunk("A", "d1", "content A", _unit_vector(DIM, 0)),
            _make_chunk("B", "d1", "content B", _scaled_unit_vector(DIM, {0: 1.0, 1: 1.0})),
            _make_chunk("C", "d1", "content C", _unit_vector(DIM, 1)),
        ]
        await store.add(chunks)

        query = _unit_vector(DIM, 0)
        results = await store.search(query, top_k=3)
        assert len(results) == 3
        # Sorted descending
        assert results[0].score >= results[1].score >= results[2].score
        # Top result is chunk A with score ~1.0
        assert results[0].chunk.id == "A"
        assert results[0].score == pytest.approx(1.0, abs=1e-5)

    async def test_search_top_k_limits_results(self) -> None:
        """top_k limits the number of returned results."""
        store = InMemoryVectorStore(dimension=DIM)
        await store.create_collection(DIM)
        chunks = [
            _make_chunk(f"c{i}", "d1", f"content {i}", _unit_vector(DIM, i))
            for i in range(5)
        ]
        await store.add(chunks)
        results = await store.search(_unit_vector(DIM, 0), top_k=2)
        assert len(results) == 2

    async def test_search_empty_store_returns_empty(self) -> None:
        """Search on an empty store returns an empty list."""
        store = InMemoryVectorStore(dimension=DIM)
        await store.create_collection(DIM)
        results = await store.search(_unit_vector(DIM, 0), top_k=5)
        assert results == []

    async def test_search_with_score_threshold_filters_low(self) -> None:
        """score_threshold filters out results below the threshold."""
        store = InMemoryVectorStore(dimension=DIM)
        await store.create_collection(DIM)
        chunks = [
            _make_chunk("A", "d1", "high score", _unit_vector(DIM, 0)),
            _make_chunk("B", "d1", "mid score", _scaled_unit_vector(DIM, {0: 1.0, 1: 1.0})),
            _make_chunk("C", "d1", "low score", _unit_vector(DIM, 1)),
        ]
        await store.add(chunks)
        # Query along axis 0: A=1.0, B~0.707, C=0.0
        # Threshold 0.5 should keep A and B, filter C
        results = await store.search(_unit_vector(DIM, 0), top_k=5, score_threshold=0.5)
        ids = {r.chunk.id for r in results}
        assert "A" in ids
        assert "B" in ids
        assert "C" not in ids

    async def test_search_with_filter_conditions_metadata(self) -> None:
        """filter_conditions filter by chunk metadata."""
        store = InMemoryVectorStore(dimension=DIM)
        await store.create_collection(DIM)
        chunks = [
            _make_chunk("c1", "d1", "tech content", _unit_vector(DIM, 0), {"category": "tech"}),
            _make_chunk("c2", "d2", "science content", _unit_vector(DIM, 0), {"category": "science"}),
            _make_chunk("c3", "d1", "more tech", _unit_vector(DIM, 1), {"category": "tech"}),
        ]
        await store.add(chunks)
        results = await store.search(
            _unit_vector(DIM, 0),
            top_k=10,
            filter_conditions={"category": "tech"},
        )
        ids = {r.chunk.id for r in results}
        assert ids == {"c1", "c3"}

    async def test_search_with_filter_conditions_document_id(self) -> None:
        """filter_conditions can filter by document_id."""
        store = InMemoryVectorStore(dimension=DIM)
        await store.create_collection(DIM)
        chunks = [
            _make_chunk("c1", "doc-A", "content 1", _unit_vector(DIM, 0)),
            _make_chunk("c2", "doc-A", "content 2", _unit_vector(DIM, 1)),
            _make_chunk("c3", "doc-B", "content 3", _unit_vector(DIM, 0)),
        ]
        await store.add(chunks)
        results = await store.search(
            _unit_vector(DIM, 0),
            top_k=10,
            filter_conditions={"document_id": "doc-A"},
        )
        ids = {r.chunk.id for r in results}
        assert ids == {"c1", "c2"}

    async def test_search_returns_search_result_objects(self) -> None:
        """Search returns SearchResult objects with correct fields."""
        store = InMemoryVectorStore(dimension=DIM)
        await store.create_collection(DIM)
        chunk = _make_chunk(
            "c1", "d1", "content", _unit_vector(DIM, 0),
            {"title": "My Doc", "source": "/path/to/doc.txt"},
        )
        await store.add([chunk])
        results = await store.search(_unit_vector(DIM, 0), top_k=1)
        assert len(results) == 1
        result = results[0]
        assert isinstance(result, SearchResult)
        assert result.chunk.id == "c1"
        assert result.source_title == "My Doc"
        assert result.source_path == "/path/to/doc.txt"


# --------------------------------------------------------------------------- #
# delete
# --------------------------------------------------------------------------- #


class TestDelete:
    """Tests for delete and delete_by_document."""

    async def test_delete_by_chunk_ids(self) -> None:
        """delete removes chunks by their IDs and returns the count deleted."""
        store = InMemoryVectorStore(dimension=DIM)
        await store.create_collection(DIM)
        chunks = [
            _make_chunk("c1", "d1", "content 1", _unit_vector(DIM, 0)),
            _make_chunk("c2", "d1", "content 2", _unit_vector(DIM, 1)),
            _make_chunk("c3", "d1", "content 3", _unit_vector(DIM, 2)),
        ]
        await store.add(chunks)
        assert await store.count() == 3
        deleted = await store.delete(["c1", "c3"])
        assert deleted == 2
        assert await store.count() == 1
        remaining = await store.search(_unit_vector(DIM, 1), top_k=5)
        assert len(remaining) == 1
        assert remaining[0].chunk.id == "c2"

    async def test_delete_nonexistent_id_returns_zero(self) -> None:
        """delete with non-existent IDs returns 0."""
        store = InMemoryVectorStore(dimension=DIM)
        await store.create_collection(DIM)
        deleted = await store.delete(["nonexistent"])
        assert deleted == 0

    async def test_delete_empty_list_returns_zero(self) -> None:
        """delete with an empty list returns 0."""
        store = InMemoryVectorStore(dimension=DIM)
        await store.create_collection(DIM)
        assert await store.delete([]) == 0

    async def test_delete_by_document(self) -> None:
        """delete_by_document removes all chunks for a document."""
        store = InMemoryVectorStore(dimension=DIM)
        await store.create_collection(DIM)
        chunks = [
            _make_chunk("c1", "doc-A", "content 1", _unit_vector(DIM, 0)),
            _make_chunk("c2", "doc-A", "content 2", _unit_vector(DIM, 1)),
            _make_chunk("c3", "doc-B", "content 3", _unit_vector(DIM, 2)),
        ]
        await store.add(chunks)
        assert await store.count() == 3
        deleted = await store.delete_by_document("doc-A")
        assert deleted == 2
        assert await store.count() == 1
        remaining = await store.search(_unit_vector(DIM, 2), top_k=5)
        assert len(remaining) == 1
        assert remaining[0].chunk.document_id == "doc-B"

    async def test_delete_by_document_nonexistent_returns_zero(self) -> None:
        """delete_by_document with a non-existent document returns 0."""
        store = InMemoryVectorStore(dimension=DIM)
        await store.create_collection(DIM)
        await store.add([_make_chunk("c1", "d1", "c", _unit_vector(DIM, 0))])
        deleted = await store.delete_by_document("nonexistent")
        assert deleted == 0


# --------------------------------------------------------------------------- #
# count
# --------------------------------------------------------------------------- #


class TestCount:
    """Tests for count."""

    async def test_count_empty_store(self) -> None:
        """count returns 0 for an empty store."""
        store = InMemoryVectorStore(dimension=DIM)
        await store.create_collection(DIM)
        assert await store.count() == 0

    async def test_count_after_add(self) -> None:
        """count returns the number of stored vectors."""
        store = InMemoryVectorStore(dimension=DIM)
        await store.create_collection(DIM)
        await store.add([
            _make_chunk("c1", "d1", "a", _unit_vector(DIM, 0)),
            _make_chunk("c2", "d1", "b", _unit_vector(DIM, 1)),
        ])
        assert await store.count() == 2

    async def test_count_after_delete(self) -> None:
        """count decreases after deletion."""
        store = InMemoryVectorStore(dimension=DIM)
        await store.create_collection(DIM)
        await store.add([
            _make_chunk("c1", "d1", "a", _unit_vector(DIM, 0)),
            _make_chunk("c2", "d1", "b", _unit_vector(DIM, 1)),
        ])
        assert await store.count() == 2
        await store.delete(["c1"])
        assert await store.count() == 1


# --------------------------------------------------------------------------- #
# health_check
# --------------------------------------------------------------------------- #


class TestHealthCheck:
    """Tests for health_check."""

    async def test_health_check_returns_true(self) -> None:
        """health_check always returns True for InMemoryVectorStore."""
        store = InMemoryVectorStore(dimension=DIM)
        await store.create_collection(DIM)
        assert await store.health_check() is True

    async def test_health_check_empty_store(self) -> None:
        """health_check returns True even for an empty store."""
        store = InMemoryVectorStore(dimension=DIM)
        assert await store.health_check() is True
