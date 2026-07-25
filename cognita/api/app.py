"""FastAPI application factory.

The :func:`create_app` function wires together every layer of the Cognita RAG
system: configuration, logging, metrics, middleware, routes, and lifecycle
hooks (startup initialisation / shutdown cleanup).
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import PlainTextResponse

from cognita.api.middleware import RateLimitMiddleware, RequestLoggingMiddleware
from cognita.api.routes import get_pipeline, router
from cognita.config import get_settings
from cognita.observability.logging import get_logger, setup_logging
from cognita.observability.metrics import init_metrics

logger = get_logger("cognita.api.app")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance.

    Performs the following steps in order:

    1.  Set up structured logging.
    2.  Initialise Prometheus metrics.
    3.  Create the FastAPI instance.
    4.  Add CORS middleware.
    5.  Add rate-limiting and request-logging middleware.
    6.  Include the API router.
    7.  Register the ``/metrics`` Prometheus endpoint.
    8.  Register startup/shutdown lifecycle hooks.
    9.  Register the root ``/`` info endpoint.
    """
    settings = get_settings()

    # 1. Set up logging.
    setup_logging()
    logger.info(
        "Initialising application",
        name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )

    # 2. Initialise metrics.
    init_metrics()

    # 3. Create the FastAPI instance.
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Production-grade RAG Knowledge Agent with hybrid retrieval, "
            "cross-encoder reranking, citation-grounded generation, and full "
            "observability."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # 4. CORS middleware.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 5. Custom middleware (added in reverse-execution order: the last
    #    add_middleware call becomes the outermost layer).
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(RequestLoggingMiddleware)

    # 6. Include the API router.
    app.include_router(router)

    # 7. Prometheus metrics endpoint.
    @app.get("/metrics", tags=["observability"], include_in_schema=False)
    async def metrics() -> PlainTextResponse:
        """Expose Prometheus metrics."""
        return PlainTextResponse(
            generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    # 8. Startup event — initialise the ingestion pipeline (creates the
    #    vector-store collection if it does not yet exist).
    @app.on_event("startup")
    async def startup() -> None:
        logger.info("Running startup tasks")
        try:
            pipeline = get_pipeline()
            await pipeline.initialize()
            logger.info("Ingestion pipeline initialised successfully")
        except Exception as exc:
            logger.error(
                "Startup initialisation failed — the API will start but "
                "ingestion/retrieval may not work until the underlying "
                "services are available",
                error=str(exc),
                exc_info=True,
            )
        logger.info("Application startup complete")

    # 9. Shutdown event — graceful cleanup.
    @app.on_event("shutdown")
    async def shutdown() -> None:
        logger.info("Running shutdown tasks")
        # Close any open resources on the LLM client.
        try:
            from cognita.core.llm import reset_llm
            from cognita.core.vectorstore import reset_vectorstore

            await reset_llm()
            await reset_vectorstore()
        except Exception as exc:
            logger.warning("Error during shutdown cleanup", error=str(exc))
        logger.info("Application shutdown complete")

    # 10. Root info endpoint.
    @app.get("/", tags=["root"])
    async def root() -> dict[str, Any]:
        """Return basic application information."""
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
            "description": "Production-grade RAG Knowledge Agent",
            "endpoints": {
                "docs": "/docs",
                "health": "/health",
                "ready": "/ready",
                "metrics": "/metrics",
                "query": "/api/v1/query",
                "stream": "/api/v1/query/stream",
                "chat_ws": "/api/v1/chat",
                "documents": "/api/v1/documents",
            },
        }

    return app
