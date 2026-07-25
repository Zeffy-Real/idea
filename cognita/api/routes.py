"""API route handlers for the Cognita RAG application.

All endpoints are registered on a single :class:`APIRouter` that is included
by the app factory in :mod:`cognita.api.app`.  Component instances (pipeline,
retriever, reranker, generator, expander) are lazily-initialised module-level
singletons accessible through ``Depends``-compatible getter functions.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from cognita.api.auth import APIKeyDependency
from cognita.api.schemas import (
    DocumentListResponse,
    DocumentResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    TextIngestRequest,
)
from cognita.config import get_settings
from cognita.core.exceptions import (
    AuthenticationError,
    CognitaError,
    DocumentLoadingError,
    GenerationError,
    RateLimitExceededError,
    RetrievalError,
    VectorStoreError,
)
from cognita.core.llm import get_llm
from cognita.core.models import Citation, DocumentStatus
from cognita.core.vectorstore import get_vectorstore
from cognita.generation.generator import RAGGenerator
from cognita.generation.memory import ConversationMemory
from cognita.ingestion.pipeline import IngestionPipeline
from cognita.observability.logging import get_logger
from cognita.observability.metrics import active_websocket_connections
from cognita.retrieval.expander import QueryExpander
from cognita.retrieval.hybrid import HybridRetriever
from cognita.retrieval.reranker import CrossEncoderReranker

logger = get_logger("cognita.api.routes")

router = APIRouter()

# --------------------------------------------------------------------------- #
# Module-level singletons (lazy init)
# --------------------------------------------------------------------------- #

_pipeline: IngestionPipeline | None = None
_retriever: HybridRetriever | None = None
_reranker: CrossEncoderReranker | None = None
_generator: RAGGenerator | None = None
_expander: QueryExpander | None = None

# Per-conversation memory store keyed by conversation_id.
_conversations: dict[str, ConversationMemory] = {}


def get_pipeline() -> IngestionPipeline:
    """Return the lazily-initialised ingestion pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline()
    return _pipeline


def get_retriever() -> HybridRetriever:
    """Return the lazily-initialised hybrid retriever singleton."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def get_reranker() -> CrossEncoderReranker:
    """Return the lazily-initialised cross-encoder reranker singleton."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker


def get_generator() -> RAGGenerator:
    """Return the lazily-initialised RAG generator singleton."""
    global _generator
    if _generator is None:
        _generator = RAGGenerator()
    return _generator


def get_expander() -> QueryExpander:
    """Return the lazily-initialised query expander singleton."""
    global _expander
    if _expander is None:
        _expander = QueryExpander()
    return _expander


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _get_conversation(conversation_id: str | None) -> ConversationMemory:
    """Get or create conversation memory for *conversation_id*."""
    settings = get_settings()
    key = conversation_id or "default"
    if key not in _conversations:
        _conversations[key] = ConversationMemory(
            max_turns=settings.conversation_memory_turns
        )
    return _conversations[key]


def _to_document_response(result: Any) -> DocumentResponse:
    """Convert an :class:`IngestionResult` to an :class:`DocumentResponse`."""
    status_value = (
        result.status.value
        if isinstance(result.status, DocumentStatus)
        else str(result.status)
    )
    return DocumentResponse(
        document_id=result.document_id,
        title=result.title,
        chunks_created=result.chunks_created,
        chunks_indexed=result.chunks_indexed,
        status=status_value,
        error=result.error,
        latency_ms=result.latency_ms,
    )


def _handle_cognita_error(exc: CognitaError) -> HTTPException:
    """Map a :class:`CognitaError` to the appropriate ``HTTPException``."""
    if isinstance(exc, AuthenticationError):
        return HTTPException(status_code=401, detail=exc.message)
    if isinstance(exc, RateLimitExceededError):
        return HTTPException(status_code=429, detail=exc.message)
    if isinstance(exc, DocumentLoadingError):
        return HTTPException(status_code=400, detail=exc.message)
    if isinstance(exc, VectorStoreError):
        return HTTPException(status_code=503, detail=exc.message)
    if isinstance(exc, (RetrievalError, GenerationError)):
        return HTTPException(status_code=500, detail=exc.message)
    return HTTPException(status_code=500, detail=exc.message)


async def _retrieve_and_rerank(
    query: str,
    history: list[Any],
    request: QueryRequest,
) -> list[Any]:
    """Shared retrieval + optional rerank logic for query endpoints."""
    settings = get_settings()
    retriever = get_retriever()

    results = await retriever.retrieve_with_context(
        query=query,
        conversation_history=history,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        filter_conditions=request.filter_conditions,
    )

    if settings.rerank_enabled and results:
        reranker = get_reranker()
        top_k = request.top_k or settings.retrieval_top_k
        results = await reranker.rerank(query, results, top_k=top_k)

    return results


# --------------------------------------------------------------------------- #
# Health & readiness
# --------------------------------------------------------------------------- #


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse:
    """Liveness probe — always returns 200 (the process is alive)."""
    settings = get_settings()

    llm = get_llm()
    vectorstore = get_vectorstore()

    llm_healthy = await llm.health_check()
    vs_healthy = await vectorstore.health_check()

    components = {
        "llm": llm_healthy,
        "vectorstore": vs_healthy,
    }
    all_healthy = all(components.values())

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        version=settings.app_version,
        components=components,
    )


@router.get("/ready", response_model=HealthResponse, tags=["health"])
async def readiness_check() -> HealthResponse:
    """Readiness probe — returns 503 when any component is unavailable."""
    settings = get_settings()

    llm = get_llm()
    vectorstore = get_vectorstore()

    llm_healthy = await llm.health_check()
    vs_healthy = await vectorstore.health_check()

    components = {
        "llm": llm_healthy,
        "vectorstore": vs_healthy,
    }

    if not all(components.values()):
        response = HealthResponse(
            status="unhealthy",
            version=settings.app_version,
            components=components,
        )
        return JSONResponse(
            status_code=503,
            content=response.model_dump(),
        )

    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        components=components,
    )


# --------------------------------------------------------------------------- #
# Document ingestion
# --------------------------------------------------------------------------- #


class _DirectoryIngestRequest(BaseModel):
    """Internal request body for directory ingestion."""

    directory: str


@router.post(
    "/api/v1/documents/text",
    response_model=DocumentResponse,
    tags=["documents"],
    summary="Ingest raw text",
)
async def ingest_text(
    request: TextIngestRequest,
    _auth: bool = Depends(APIKeyDependency),
) -> DocumentResponse:
    """Ingest a raw text document into the knowledge base."""
    try:
        pipeline = get_pipeline()
        result = await pipeline.ingest_text(
            title=request.title,
            content=request.content,
            source=request.source,
        )
        return _to_document_response(result)
    except CognitaError as exc:
        raise _handle_cognita_error(exc) from exc
    except Exception as exc:
        logger.error("Text ingestion failed", error=str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post(
    "/api/v1/documents",
    response_model=DocumentResponse,
    tags=["documents"],
    summary="Upload and ingest a file",
)
async def upload_document(
    file: UploadFile = File(...),
    _auth: bool = Depends(APIKeyDependency),
) -> DocumentResponse:
    """Upload a file and ingest it into the knowledge base."""
    tmp_path: str | None = None
    try:
        # Preserve the original extension so the loader can pick the right
        # parser.
        original_name = file.filename or "upload"
        suffix = os.path.splitext(original_name)[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        pipeline = get_pipeline()
        result = await pipeline.ingest_file(tmp_path)
        return _to_document_response(result)
    except CognitaError as exc:
        raise _handle_cognita_error(exc) from exc
    except Exception as exc:
        logger.error("File upload ingestion failed", error=str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                logger.warning("Failed to delete temp file", path=tmp_path)


@router.post(
    "/api/v1/documents/directory",
    response_model=list[DocumentResponse],
    tags=["documents"],
    summary="Ingest all files in a directory",
)
async def ingest_directory(
    request: _DirectoryIngestRequest,
    _auth: bool = Depends(APIKeyDependency),
) -> list[DocumentResponse]:
    """Ingest every supported file beneath the given directory path."""
    try:
        pipeline = get_pipeline()
        results = await pipeline.ingest_directory(request.directory)
        return [_to_document_response(r) for r in results]
    except CognitaError as exc:
        raise _handle_cognita_error(exc) from exc
    except Exception as exc:
        logger.error("Directory ingestion failed", error=str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/api/v1/documents",
    response_model=DocumentListResponse,
    tags=["documents"],
    summary="Get document statistics",
)
async def get_documents(
    _auth: bool = Depends(APIKeyDependency),
) -> DocumentListResponse:
    """Return the total number of chunks currently stored."""
    try:
        vectorstore = get_vectorstore()
        total = await vectorstore.count()
        return DocumentListResponse(
            total_chunks=total,
            components={
                "total_chunks": total,
                "collection": get_settings().qdrant_collection,
            },
        )
    except CognitaError as exc:
        raise _handle_cognita_error(exc) from exc
    except Exception as exc:
        logger.error("Failed to get document stats", error=str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete(
    "/api/v1/documents/{document_id}",
    tags=["documents"],
    summary="Delete a document and all its chunks",
)
async def delete_document(
    document_id: str,
    _auth: bool = Depends(APIKeyDependency),
) -> dict[str, Any]:
    """Delete a document and every chunk associated with it."""
    try:
        vectorstore = get_vectorstore()
        deleted = await vectorstore.delete_by_document(document_id)
        return {
            "document_id": document_id,
            "chunks_deleted": deleted,
            "status": "deleted",
        }
    except CognitaError as exc:
        raise _handle_cognita_error(exc) from exc
    except Exception as exc:
        logger.error("Document deletion failed", error=str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# --------------------------------------------------------------------------- #
# Query endpoints
# --------------------------------------------------------------------------- #


@router.post(
    "/api/v1/query",
    response_model=QueryResponse,
    tags=["query"],
    summary="Query the knowledge base",
)
async def query(
    request: QueryRequest,
    _auth: bool = Depends(APIKeyDependency),
) -> QueryResponse:
    """Retrieve relevant context and generate a grounded answer."""
    try:
        settings = get_settings()

        # Get conversation memory (prior turns only).
        memory = _get_conversation(request.conversation_id)
        history = memory.get_messages()

        # Optional query expansion.
        query_text = request.query
        if request.expand_query:
            expander = get_expander()
            variations = await expander.expand(query_text)
            # Use the original query for generation; variations improve
            # retrieval recall when folded into the contextual query below.
            logger.debug(
                "Query expanded",
                variations=len(variations),
                original=query_text,
            )

        # Retrieve + optional rerank.
        results = await _retrieve_and_rerank(query_text, history, request)

        # Generate the answer.
        generator = get_generator()
        gen_response = await generator.generate(
            query=query_text,
            search_results=results,
            conversation_history=history,
            thinking=request.thinking,
        )

        # Update conversation memory with the completed exchange.
        memory.add_user_message(query_text)
        memory.add_assistant_message(gen_response.answer, gen_response.citations)

        return QueryResponse(
            answer=gen_response.answer,
            citations=gen_response.citations,
            usage=gen_response.usage,
            model=gen_response.model,
            thinking=gen_response.thinking,
            latency_ms=gen_response.latency_ms,
        )
    except CognitaError as exc:
        raise _handle_cognita_error(exc) from exc
    except Exception as exc:
        logger.error("Query failed", error=str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post(
    "/api/v1/query/stream",
    tags=["query"],
    summary="Stream a query response (SSE)",
)
async def stream_query(
    request: QueryRequest,
    _auth: bool = Depends(APIKeyDependency),
) -> StreamingResponse:
    """Stream the generated answer token-by-token using Server-Sent Events."""
    settings = get_settings()

    async def event_generator():
        try:
            # Get conversation memory (prior turns only).
            memory = _get_conversation(request.conversation_id)
            history = memory.get_messages()

            # Retrieve + optional rerank.
            results = await _retrieve_and_rerank(request.query, history, request)

            # Stream generation tokens.
            generator = get_generator()
            collected_tokens: list[str] = []

            async for token in generator.generate_stream(
                query=request.query,
                search_results=results,
                conversation_history=history,
                thinking=request.thinking,
            ):
                collected_tokens.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"

            # Build citations from the final result set.
            citations = [
                Citation.from_search_result(r).model_dump() for r in results
            ]

            # Send a terminal event with citations and usage info.
            yield f"data: {json.dumps({'done': True, 'citations': citations})}\n\n"

            # Update conversation memory.
            full_answer = "".join(collected_tokens)
            memory.add_user_message(request.query)
            memory.add_assistant_message(
                full_answer,
                [Citation.from_search_result(r) for r in results],
            )

        except CognitaError as exc:
            logger.error("Stream query failed", error=str(exc), exc_info=True)
            yield f"data: {json.dumps({'error': exc.message, 'details': exc.details})}\n\n"
        except Exception as exc:
            logger.error("Stream query failed", error=str(exc), exc_info=True)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


# --------------------------------------------------------------------------- #
# WebSocket chat
# --------------------------------------------------------------------------- #


@router.websocket("/api/v1/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    """WebSocket endpoint for interactive streaming chat.

    Accepts JSON messages of the form::

        {"query": "...", "conversation_id": "...", "thinking": true}

    and streams back token events followed by a terminal ``done`` event
    containing the full answer and citations.
    """
    await websocket.accept()
    active_websocket_connections.inc()

    try:
        while True:
            # Receive a message from the client.
            try:
                data = await websocket.receive_json()
            except Exception as exc:
                logger.warning("WebSocket received non-JSON data", error=str(exc))
                await websocket.send_json({"type": "error", "error": "Invalid JSON message"})
                continue

            query = data.get("query", "")
            if not query or not query.strip():
                await websocket.send_json({"type": "error", "error": "Query is required"})
                continue

            conversation_id = data.get("conversation_id")
            thinking = data.get("thinking")
            top_k = data.get("top_k")
            score_threshold = data.get("score_threshold")
            filter_conditions = data.get("filter_conditions")

            try:
                settings = get_settings()

                # Get conversation memory (prior turns only).
                memory = _get_conversation(conversation_id)
                history = memory.get_messages()

                # Retrieve + optional rerank.
                retriever = get_retriever()
                results = await retriever.retrieve_with_context(
                    query=query,
                    conversation_history=history,
                    top_k=top_k,
                    score_threshold=score_threshold,
                    filter_conditions=filter_conditions,
                )

                if settings.rerank_enabled and results:
                    reranker = get_reranker()
                    effective_top_k = top_k or settings.retrieval_top_k
                    results = await reranker.rerank(query, results, top_k=effective_top_k)

                # Stream generation tokens.
                generator = get_generator()
                collected_tokens: list[str] = []

                async for token in generator.generate_stream(
                    query=query,
                    search_results=results,
                    conversation_history=history,
                    thinking=thinking,
                ):
                    await websocket.send_json({"type": "token", "token": token})
                    collected_tokens.append(token)

                # Build citations.
                citations = [
                    Citation.from_search_result(r).model_dump() for r in results
                ]

                # Send the terminal event.
                full_answer = "".join(collected_tokens)
                await websocket.send_json({
                    "type": "done",
                    "answer": full_answer,
                    "citations": citations,
                })

                # Update conversation memory.
                memory.add_user_message(query)
                memory.add_assistant_message(
                    full_answer,
                    [Citation.from_search_result(r) for r in results],
                )

            except CognitaError as exc:
                logger.error("WebSocket chat error", error=str(exc), exc_info=True)
                await websocket.send_json({
                    "type": "error",
                    "error": exc.message,
                    "details": exc.details,
                })
            except Exception as exc:
                logger.error("WebSocket chat error", error=str(exc), exc_info=True)
                await websocket.send_json({"type": "error", "error": str(exc)})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as exc:
        logger.error("WebSocket error", error=str(exc), exc_info=True)
    finally:
        active_websocket_connections.dec()
