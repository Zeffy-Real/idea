"""Shared pytest fixtures and configuration for the Cognita RAG test suite.

Environment variables are set at module level — BEFORE any cognita modules
are imported — so that ``pydantic-settings`` picks up the test configuration
on first load.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Set environment variables BEFORE importing cognita modules.
# --------------------------------------------------------------------------- #
import os

os.environ["DEEPSEEK_API_KEY"] = "test-key"
os.environ["VECTOR_STORE_TYPE"] = "memory"
os.environ["ENVIRONMENT"] = "development"

# --------------------------------------------------------------------------- #
# Now it is safe to import cognita modules.
# --------------------------------------------------------------------------- #
import random
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from cognita.config import Settings, reload_settings
from cognita.core.models import Chunk, SearchResult
from cognita.core.vectorstore import InMemoryVectorStore


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def settings() -> Settings:
    """Return a Settings instance with test configuration and reload the cache."""
    os.environ["DEEPSEEK_API_KEY"] = "test-key"
    os.environ["VECTOR_STORE_TYPE"] = "memory"
    os.environ["ENVIRONMENT"] = "development"
    os.environ["API_KEY"] = ""
    return reload_settings()


@pytest_asyncio.fixture
async def vectorstore():
    """Create an InMemoryVectorStore for testing, reset after use."""
    store = InMemoryVectorStore(dimension=512)
    await store.create_collection(512)
    yield store
    # Clear the module-level singleton so subsequent tests start clean.
    import cognita.core.vectorstore as vs_module

    vs_module._vectorstore_instance = None


@pytest.fixture
def sample_text() -> str:
    """Return a multi-paragraph text about technology."""
    return (
        "Artificial intelligence is transforming the way we interact with "
        "technology. From voice assistants to autonomous vehicles, AI is "
        "becoming ubiquitous in modern society.\n\n"
        "Machine learning, a subset of AI, enables systems to learn from data "
        "without explicit programming. Deep learning, in turn, uses neural "
        "networks with many layers to model complex patterns in large "
        "datasets.\n\n"
        "Natural language processing allows computers to understand human "
        "language. Large language models like GPT and BERT have revolutionized "
        "this field, enabling sophisticated text generation and comprehension "
        "capabilities.\n\n"
        "Computer vision gives machines the ability to see and interpret "
        "visual data. Convolutional neural networks are the backbone of most "
        "modern vision systems, powering applications from facial recognition "
        "to medical imaging."
    )


def _random_normalized_vector(dim: int = 512) -> list[float]:
    """Generate a random L2-normalized vector of the given dimension."""
    vec = [random.random() for _ in range(dim)]
    magnitude = sum(v * v for v in vec) ** 0.5
    if magnitude == 0:
        return vec
    return [v / magnitude for v in vec]


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Return a list of 3 Chunk objects with random normalized embeddings."""
    contents = [
        "Artificial intelligence is transforming the way we interact with technology.",
        "Machine learning enables systems to learn from data without explicit programming.",
        "Natural language processing allows computers to understand human language.",
    ]
    chunks: list[Chunk] = []
    for i, content in enumerate(contents):
        chunk = Chunk(
            document_id="test-doc-001",
            content=content,
            index=i,
            token_count=len(content.split()),
            metadata={
                "title": "AI Overview",
                "source": "test_source.txt",
                "file_type": "txt",
            },
            embedding=_random_normalized_vector(512),
        )
        chunks.append(chunk)
    return chunks


@pytest.fixture
def sample_search_results(sample_chunks: list[Chunk]) -> list[SearchResult]:
    """Return a list of 3 SearchResult objects derived from sample_chunks."""
    return [
        SearchResult(
            chunk=chunk,
            score=0.95 - i * 0.1,
            source_title="AI Overview",
            source_path="test_source.txt",
        )
        for i, chunk in enumerate(sample_chunks)
    ]


@pytest.fixture
def mock_llm() -> AsyncMock:
    """Return an AsyncMock that mimics DeepSeekLLM.

    ``chat`` returns a tuple of (content, usage_dict, thinking_content) matching
    the real DeepSeekLLM.chat return signature.
    """
    mock = AsyncMock()
    mock.chat.return_value = (
        "test answer",
        {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        None,
    )
    mock.health_check.return_value = True
    mock.close = AsyncMock()
    return mock
