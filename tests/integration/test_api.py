"""Integration tests for the FastAPI API.

All external dependencies (DeepSeek LLM, sentence-transformers embedding,
Qdrant) are mocked so the tests run fully offline.
"""

from __future__ import annotations

# Environment variables must be set before importing cognita modules.
# conftest.py already sets them at module level, but we reinforce here.
import os

os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("VECTOR_STORE_TYPE", "memory")
os.environ.setdefault("ENVIRONMENT", "development")
# Explicitly disable API key auth by default so the .env file value does not
# accidentally enable authentication in the no-auth fixture.
os.environ.setdefault("API_KEY", "")

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

from cognita.api import routes as routes_mod
from cognita.api.app import create_app
from cognita.config import reload_settings
from cognita.core.models import DocumentStatus, GenerationResponse, IngestionResult
from cognita.core.vectorstore import InMemoryVectorStore


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _reset_singletons() -> None:
    """Reset all module-level singletons in routes.py."""
    routes_mod._pipeline = None
    routes_mod._retriever = None
    routes_mod._reranker = None
    routes_mod._generator = None
    routes_mod._expander = None
    routes_mod._conversations.clear()


def _make_mock_llm() -> AsyncMock:
    """Create a mock LLM with health_check and chat."""
    mock = AsyncMock()
    mock.health_check = AsyncMock(return_value=True)
    mock.chat = AsyncMock(
        return_value=(
            "This is a test answer grounded in the provided context.",
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            None,
        )
    )
    mock.close = AsyncMock()
    return mock


def _make_mock_pipeline() -> AsyncMock:
    """Create a mock ingestion pipeline."""
    mock = AsyncMock()
    mock.initialize = AsyncMock(return_value=None)
    mock.ingest_text = AsyncMock(
        return_value=IngestionResult(
            document_id="test-doc-id",
            title="Test Document",
            chunks_created=3,
            chunks_indexed=3,
            status=DocumentStatus.INDEXED,
            latency_ms=42.0,
        )
    )
    return mock


def _make_mock_retriever() -> AsyncMock:
    """Create a mock hybrid retriever."""
    mock = AsyncMock()
    mock.retrieve_with_context = AsyncMock(return_value=[])
    return mock


def _make_mock_generator() -> AsyncMock:
    """Create a mock RAG generator."""
    mock = AsyncMock()
    mock.generate = AsyncMock(
        return_value=GenerationResponse(
            answer="Test answer based on retrieved context.",
            citations=[],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            model="deepseek-chat",
            latency_ms=100.0,
        )
    )
    return mock


def _make_mock_reranker() -> AsyncMock:
    """Create a mock reranker that passes results through unchanged."""
    mock = AsyncMock()

    async def _passthrough(query: str, results: list[Any], **kw: Any) -> list[Any]:
        return results

    mock.rerank = AsyncMock(side_effect=_passthrough)
    return mock


def _make_mock_expander() -> AsyncMock:
    """Create a mock query expander."""
    mock = AsyncMock()
    mock.expand = AsyncMock(return_value=[])
    return mock


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def app_client() -> Any:
    """Yield a TestClient with all external dependencies mocked (no auth)."""
    # Ensure auth is disabled — override the .env file value.
    os.environ["API_KEY"] = ""
    _reset_singletons()
    reload_settings()

    mock_llm = _make_mock_llm()
    mock_pipeline = _make_mock_pipeline()
    mock_retriever = _make_mock_retriever()
    mock_generator = _make_mock_generator()
    mock_reranker = _make_mock_reranker()
    mock_expander = _make_mock_expander()
    real_vectorstore = InMemoryVectorStore(dimension=512)

    import cognita.api.app as app_mod

    with (
        patch.object(routes_mod, "get_llm", return_value=mock_llm),
        patch.object(routes_mod, "get_vectorstore", return_value=real_vectorstore),
        patch.object(routes_mod, "get_pipeline", return_value=mock_pipeline),
        patch.object(routes_mod, "get_retriever", return_value=mock_retriever),
        patch.object(routes_mod, "get_generator", return_value=mock_generator),
        patch.object(routes_mod, "get_reranker", return_value=mock_reranker),
        patch.object(routes_mod, "get_expander", return_value=mock_expander),
        patch.object(app_mod, "get_pipeline", return_value=mock_pipeline),
    ):
        app = create_app()
        with TestClient(app) as client:
            yield client

    _reset_singletons()


@pytest.fixture
def app_client_with_auth() -> Any:
    """Yield a TestClient with authentication enabled (API_KEY=test-secret)."""
    os.environ["API_KEY"] = "test-secret"
    _reset_singletons()
    reload_settings()

    mock_llm = _make_mock_llm()
    mock_pipeline = _make_mock_pipeline()
    mock_retriever = _make_mock_retriever()
    mock_generator = _make_mock_generator()
    mock_reranker = _make_mock_reranker()
    mock_expander = _make_mock_expander()
    real_vectorstore = InMemoryVectorStore(dimension=512)

    import cognita.api.app as app_mod

    with (
        patch.object(routes_mod, "get_llm", return_value=mock_llm),
        patch.object(routes_mod, "get_vectorstore", return_value=real_vectorstore),
        patch.object(routes_mod, "get_pipeline", return_value=mock_pipeline),
        patch.object(routes_mod, "get_retriever", return_value=mock_retriever),
        patch.object(routes_mod, "get_generator", return_value=mock_generator),
        patch.object(routes_mod, "get_reranker", return_value=mock_reranker),
        patch.object(routes_mod, "get_expander", return_value=mock_expander),
        patch.object(app_mod, "get_pipeline", return_value=mock_pipeline),
    ):
        app = create_app()
        with TestClient(app) as client:
            yield client

    _reset_singletons()
    # Restore no-auth environment — set to empty (not pop) so it overrides
    # the .env file value which may be parsed as a non-empty comment string.
    os.environ["API_KEY"] = ""
    reload_settings()


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


@pytest.mark.integration
class TestHealthEndpoint:
    """GET /health"""

    def test_health_returns_200(self, app_client: TestClient) -> None:
        """GET /health returns HTTP 200."""
        response = app_client.get("/health")
        assert response.status_code == 200

    def test_health_contains_status_field(self, app_client: TestClient) -> None:
        """GET /health response contains a 'status' field."""
        response = app_client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] in ("healthy", "degraded")

    def test_health_contains_version(self, app_client: TestClient) -> None:
        """GET /health response contains a 'version' field."""
        response = app_client.get("/health")
        data = response.json()
        assert "version" in data

    def test_health_contains_components(self, app_client: TestClient) -> None:
        """GET /health response contains a 'components' dict."""
        response = app_client.get("/health")
        data = response.json()
        assert "components" in data
        assert "llm" in data["components"]
        assert "vectorstore" in data["components"]


@pytest.mark.integration
class TestRootEndpoint:
    """GET /"""

    def test_root_returns_200(self, app_client: TestClient) -> None:
        """GET / returns HTTP 200."""
        response = app_client.get("/")
        assert response.status_code == 200

    def test_root_contains_app_info(self, app_client: TestClient) -> None:
        """GET / returns application information."""
        response = app_client.get("/")
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "environment" in data
        assert "endpoints" in data

    def test_root_lists_endpoints(self, app_client: TestClient) -> None:
        """GET / lists available endpoints."""
        response = app_client.get("/")
        data = response.json()
        endpoints = data["endpoints"]
        assert "health" in endpoints
        assert "query" in endpoints
        assert "documents" in endpoints


@pytest.mark.integration
class TestTextIngestion:
    """POST /api/v1/documents/text"""

    def test_ingest_text_returns_200(self, app_client: TestClient) -> None:
        """POST /api/v1/documents/text with valid text returns 200."""
        response = app_client.post(
            "/api/v1/documents/text",
            json={
                "title": "Test Document",
                "content": "This is a test document about artificial intelligence.",
                "source": "test_source",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Test Document"
        assert data["status"] == "indexed"
        assert data["chunks_created"] > 0

    def test_ingest_text_missing_title_returns_422(self, app_client: TestClient) -> None:
        """POST /api/v1/documents/text without title returns 422."""
        response = app_client.post(
            "/api/v1/documents/text",
            json={"content": "some content"},
        )
        assert response.status_code == 422

    def test_ingest_text_missing_content_returns_422(self, app_client: TestClient) -> None:
        """POST /api/v1/documents/text without content returns 422."""
        response = app_client.post(
            "/api/v1/documents/text",
            json={"title": "Title only"},
        )
        assert response.status_code == 422

    def test_ingest_text_with_auth_no_key_returns_401(
        self, app_client_with_auth: TestClient
    ) -> None:
        """POST /api/v1/documents/text without API key returns 401 when auth is enabled."""
        response = app_client_with_auth.post(
            "/api/v1/documents/text",
            json={"title": "T", "content": "C"},
        )
        assert response.status_code == 401

    def test_ingest_text_with_auth_valid_key_returns_200(
        self, app_client_with_auth: TestClient
    ) -> None:
        """POST /api/v1/documents/text with valid API key returns 200."""
        response = app_client_with_auth.post(
            "/api/v1/documents/text",
            json={"title": "T", "content": "C"},
            headers={"X-API-Key": "test-secret"},
        )
        assert response.status_code == 200

    def test_ingest_text_with_auth_bearer_token_returns_200(
        self, app_client_with_auth: TestClient
    ) -> None:
        """POST /api/v1/documents/text with Bearer token returns 200."""
        response = app_client_with_auth.post(
            "/api/v1/documents/text",
            json={"title": "T", "content": "C"},
            headers={"Authorization": "Bearer test-secret"},
        )
        assert response.status_code == 200


@pytest.mark.integration
class TestQueryEndpoint:
    """POST /api/v1/query"""

    def test_query_returns_200(self, app_client: TestClient) -> None:
        """POST /api/v1/query returns 200 with a generated answer."""
        response = app_client.post(
            "/api/v1/query",
            json={"query": "What is artificial intelligence?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert len(data["answer"]) > 0
        assert "model" in data
        assert "latency_ms" in data

    def test_query_empty_query_returns_422(self, app_client: TestClient) -> None:
        """POST /api/v1/query with empty query returns 422."""
        response = app_client.post(
            "/api/v1/query",
            json={"query": ""},
        )
        assert response.status_code == 422

    def test_query_response_contains_usage(self, app_client: TestClient) -> None:
        """POST /api/v1/query response contains token usage info."""
        response = app_client.post(
            "/api/v1/query",
            json={"query": "What is ML?"},
        )
        data = response.json()
        assert "usage" in data

    def test_query_with_top_k(self, app_client: TestClient) -> None:
        """POST /api/v1/query accepts top_k parameter."""
        response = app_client.post(
            "/api/v1/query",
            json={"query": "What is AI?", "top_k": 3},
        )
        assert response.status_code == 200


@pytest.mark.integration
class TestMetricsEndpoint:
    """GET /metrics"""

    def test_metrics_returns_200(self, app_client: TestClient) -> None:
        """GET /metrics returns HTTP 200."""
        response = app_client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_returns_text_format(self, app_client: TestClient) -> None:
        """GET /metrics returns Prometheus text format."""
        response = app_client.get("/metrics")
        content_type = response.headers.get("content-type", "")
        assert "text/plain" in content_type

    def test_metrics_contains_cognita_metrics(self, app_client: TestClient) -> None:
        """GET /metrics response body contains cognita_ prefixed metrics."""
        # Make a request first to generate some metrics.
        app_client.get("/")
        response = app_client.get("/metrics")
        body = response.text
        assert "cognita_" in body


@pytest.mark.integration
class TestReadyEndpoint:
    """GET /ready"""

    def test_ready_returns_200_when_healthy(self, app_client: TestClient) -> None:
        """GET /ready returns 200 when all components are healthy."""
        response = app_client.get("/ready")
        # With mocked LLM (health_check=True) and InMemoryVectorStore (health_check=True),
        # the readiness check should return 200.
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


@pytest.mark.integration
class TestDocumentsListEndpoint:
    """GET /api/v1/documents"""

    def test_get_documents_returns_200(self, app_client: TestClient) -> None:
        """GET /api/v1/documents returns 200 with document stats."""
        response = app_client.get("/api/v1/documents")
        assert response.status_code == 200
        data = response.json()
        assert "total_chunks" in data
        assert isinstance(data["total_chunks"], int)
