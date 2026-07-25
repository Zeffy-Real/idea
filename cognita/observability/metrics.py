"""
Prometheus metrics collection for monitoring application behavior.
Tracks request latency, token usage, retrieval quality, and error rates.
"""

from __future__ import annotations

from typing import Any

from prometheus_client import Counter, Gauge, Histogram, Info

from cognita.config import get_settings

_METRICS_PREFIX = "cognita"

# Application info
app_info = Info(f"{_METRICS_PREFIX}_app", "Application information")

# LLM metrics
llm_requests_total = Counter(
    f"{_METRICS_PREFIX}_llm_requests_total",
    "Total LLM API requests",
    ["model", "status"],
)
llm_request_duration = Histogram(
    f"{_METRICS_PREFIX}_llm_request_duration_seconds",
    "LLM API request duration in seconds",
    ["model"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)
llm_tokens_used = Counter(
    f"{_METRICS_PREFIX}_llm_tokens_total",
    "Total tokens used by the LLM",
    ["model", "type"],  # type: prompt | completion
)

# Embedding metrics
embedding_requests_total = Counter(
    f"{_METRICS_PREFIX}_embedding_requests_total",
    "Total embedding requests",
    ["status"],
)
embedding_request_duration = Histogram(
    f"{_METRICS_PREFIX}_embedding_duration_seconds",
    "Embedding request duration in seconds",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# Retrieval metrics
retrieval_requests_total = Counter(
    f"{_METRICS_PREFIX}_retrieval_requests_total",
    "Total retrieval requests",
    ["status"],
)
retrieval_duration = Histogram(
    f"{_METRICS_PREFIX}_retrieval_duration_seconds",
    "Retrieval request duration in seconds",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
retrieval_results_count = Histogram(
    f"{_METRICS_PREFIX}_retrieval_results_count",
    "Number of results returned per retrieval",
    buckets=(1, 2, 3, 5, 8, 10, 15, 20),
)
retrieval_score = Histogram(
    f"{_METRICS_PREFIX}_retrieval_score",
    "Similarity scores of retrieved documents",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# Ingestion metrics
ingestion_requests_total = Counter(
    f"{_METRICS_PREFIX}_ingestion_requests_total",
    "Total document ingestion requests",
    ["file_type", "status"],
)
ingestion_duration = Histogram(
    f"{_METRICS_PREFIX}_ingestion_duration_seconds",
    "Document ingestion duration in seconds",
    ["file_type"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)
ingestion_chunks_created = Counter(
    f"{_METRICS_PREFIX}_ingestion_chunks_total",
    "Total chunks created during ingestion",
)

# API metrics
api_requests_total = Counter(
    f"{_METRICS_PREFIX}_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"],
)
api_request_duration = Histogram(
    f"{_METRICS_PREFIX}_api_request_duration_seconds",
    "API request duration in seconds",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# Vector store metrics
vectorstore_operations_total = Counter(
    f"{_METRICS_PREFIX}_vectorstore_operations_total",
    "Total vector store operations",
    ["operation", "status"],
)
vectorstore_collection_size = Gauge(
    f"{_METRICS_PREFIX}_vectorstore_collection_size",
    "Number of vectors in the collection",
)

# Active connections
active_websocket_connections = Gauge(
    f"{_METRICS_PREFIX}_active_websocket_connections",
    "Number of active WebSocket connections",
)

# Error metrics
errors_total = Counter(
    f"{_METRICS_PREFIX}_errors_total",
    "Total application errors",
    ["type", "severity"],
)


def init_metrics() -> None:
    """Initialize metrics with current application info."""
    settings = get_settings()
    app_info.info(
        {
            "name": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
        }
    )


def record_error(error_type: str, severity: str = "error") -> None:
    """Record an application error."""
    errors_total.labels(type=error_type, severity=severity).inc()


class MetricsTimer:
    """Context manager for timing operations and recording to a histogram."""

    def __init__(self, histogram: Histogram, labels: dict[str, str] | None = None):
        self._histogram = histogram
        self._labels = labels or {}
        self._timer = None

    def __enter__(self) -> MetricsTimer:
        if self._labels:
            self._timer = self._histogram.labels(**self._labels).time()
        else:
            self._timer = self._histogram.time()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._timer:
            self._timer.__exit__(exc_type, exc_val, exc_tb)
