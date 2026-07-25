"""Custom ASGI middleware for rate limiting and request logging.

  * :class:`RateLimitMiddleware` — simple sliding-window rate limiter keyed
    on the client's IP address.
  * :class:`RequestLoggingMiddleware` — structured per-request logging and
    Prometheus metric recording.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from cognita.config import get_settings
from cognita.observability.logging import get_logger
from cognita.observability.metrics import api_request_duration, api_requests_total

logger = get_logger("cognita.api.middleware")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter keyed on client IP.

    Uses an in-memory ``dict`` of timestamps per IP.  Expired entries are
    pruned on every request so the dictionary does not grow unbounded for
    active clients.  Inactive clients' entries are lazily cleaned on
    subsequent requests.
    """

    def __init__(self, app: Any) -> None:
        super().__init__(app)
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        settings = get_settings()

        # Bypass entirely when rate limiting is disabled.
        if not settings.rate_limit_enabled:
            return await call_next(request)

        # Determine the client IP (fall back to "unknown" for edge cases).
        client_ip = request.client.host if request.client else "unknown"

        now = time.monotonic()
        window = settings.rate_limit_window
        max_requests = settings.rate_limit_requests

        async with self._lock:
            timestamps = self._requests[client_ip]

            # Prune entries that fall outside the sliding window.
            cutoff = now - window
            self._requests[client_ip] = [t for t in timestamps if t > cutoff]

            if len(self._requests[client_ip]) >= max_requests:
                logger.warning(
                    "Rate limit exceeded",
                    client_ip=client_ip,
                    requests=len(self._requests[client_ip]),
                    limit=max_requests,
                    window=window,
                )
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "details": {
                            "retry_after": window,
                            "limit": max_requests,
                            "window_seconds": window,
                        },
                    },
                    headers={"Retry-After": str(window)},
                )

            # Record this request's timestamp.
            self._requests[client_ip].append(now)

        return await call_next(request)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs every request and records Prometheus metrics.

    Captures the HTTP method, path, status code, and duration for each
    request.  Latency is recorded into the ``api_request_duration``
    histogram and the total counter is incremented.
    """

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        method = request.method
        path = request.url.path

        # Skip logging/metrics for the metrics endpoint itself to avoid noise.
        skip_metrics = path == "/metrics"

        start = time.perf_counter()

        try:
            response = await call_next(request)
            duration = time.perf_counter() - start
            status_code = response.status_code

            if not skip_metrics:
                logger.info(
                    "Request completed",
                    method=method,
                    path=path,
                    status=status_code,
                    duration_ms=round(duration * 1000, 2),
                )
                api_requests_total.labels(
                    method=method,
                    endpoint=path,
                    status=str(status_code),
                ).inc()
                api_request_duration.labels(
                    method=method,
                    endpoint=path,
                ).observe(duration)

            return response

        except Exception as exc:
            duration = time.perf_counter() - start
            logger.error(
                "Request failed",
                method=method,
                path=path,
                error=str(exc),
                duration_ms=round(duration * 1000, 2),
                exc_info=True,
            )
            if not skip_metrics:
                api_requests_total.labels(
                    method=method,
                    endpoint=path,
                    status="500",
                ).inc()
                api_request_duration.labels(
                    method=method,
                    endpoint=path,
                ).observe(duration)
            raise
