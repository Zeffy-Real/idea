# Cognita RAG

> Production-grade Retrieval-Augmented Generation Knowledge Agent

[![CI/CD](https://github.com/ZeffyTheCoder/cognita-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/ZeffyTheCoder/cognita-rag/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://docs.astral.sh/ruff/)

Cognita RAG is a fully-featured, production-ready RAG system that ingests documents, builds a searchable vector index, and answers questions with citation-grounded responses. It is designed for real-world deployment with proper error handling, observability, authentication, rate limiting, and containerization.

---

## Table of Contents

- [Architecture](#architecture)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [CLI Usage](#cli-usage)
- [Testing](#testing)
- [Docker Deployment](#docker-deployment)
- [CI/CD Pipeline](#cicd-pipeline)
- [Observability](#observability)
- [Contributing](#contributing)
- [License](#license)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Client Layer                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐   │
│  │  REST    │  │  SSE     │  │  WebSocket   │  │  CLI (Rich TUI)   │   │
│  │  /query  │  │  /stream │  │  /chat       │  │  cognita chat     │   │
│  └────┬─────┘  └────┬─────┘  └──────┬───────┘  └────────┬──────────┘   │
└───────┼─────────────┼───────────────┼───────────────────┼──────────────┘
        │             │               │                   │
┌───────┼─────────────┼───────────────┼───────────────────┼──────────────┐
│       ▼             ▼               ▼                   ▼              │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    FastAPI Application                           │   │
│  │  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────────┐   │   │
│  │  │  CORS   │  │ Rate     │  │ Request  │  │  API Key Auth   │   │   │
│  │  │  Mid-   │  │ Limit    │  │ Logging  │  │  (Header/Bearer)│   │   │
│  │  │  ware   │  │ Mid-ware │  │ Mid-ware │  │                 │   │   │
│  │  └─────────┘  └──────────┘  └──────────┘  └─────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                         API Layer                                        │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌────────────────────┐
│  Ingestion      │    │  Retrieval       │    │  Generation        │
│  Pipeline       │    │  Engine          │    │  Engine            │
│                 │    │                  │    │                    │
│  ┌───────────┐  │    │  ┌────────────┐  │    │  ┌──────────────┐  │
│  │  Loaders  │  │    │  │  Hybrid    │  │    │  │  Prompt      │  │
│  │  PDF/MD/  │  │    │  │  Retriever │  │    │  │  Builder     │  │
│  │  TXT/DOCX │  │    │  │  (2x over- │  │    │  │  (citation   │  │
│  └─────┬─────┘  │    │  │  fetch)    │  │    │  │  enforcement)│  │
│        ▼        │    │  └─────┬──────┘  │    │  └──────┬───────┘  │
│  ┌───────────┐  │    │        ▼         │    │         ▼          │
│  │  Chunker  │  │    │  ┌────────────┐  │    │  ┌──────────────┐  │
│  │  (token-  │  │    │  │  Cross-    │  │    │  │  RAG         │  │
│  │  aware,   │  │    │  │  Encoder   │  │    │  │  Generator   │  │
│  │  recursive)│  │    │  │  Reranker  │  │    │  │  (stream +   │  │
│  └─────┬─────┘  │    │  └────────────┘  │    │  │  sync)       │  │
│        ▼        │    │                  │    │  └──────┬───────┘  │
│  ┌───────────┐  │    │  ┌────────────┐  │    │         ▼          │
│  │  Embedder │  │    │  │  Query     │  │    │  ┌──────────────┐  │
│  │  (BGE-    │  │    │  │  Expander  │  │    │  │  DeepSeek    │  │
│  │  small-zh │  │    │  │  (HyDE +   │  │    │  │  LLM         │  │
│  │  v1.5)    │  │    │  │  variants) │  │    │  │  (v4-flash / │  │
│  └─────┬─────┘  │    │  └────────────┘  │    │  │  v4-pro)     │  │
│        ▼        │    └──────────────────┘    │  └──────────────┘  │
│  ┌───────────┐  │                           │                    │
│  │  Vector   │  │                           │  ┌──────────────┐  │
│  │  Store    │◄─┼───────────────────────────┼─│  Conversation │  │
│  │  (Qdrant) │  │                           │  │  Memory      │  │
│  └───────────┘  │                           │  └──────────────┘  │
└─────────────────┘                           └────────────────────┘
         │                                              │
         ▼                                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Observability Layer                             │
│  ┌──────────────┐  ┌─────────────────┐  ┌──────────────────────┐   │
│  │  Structured  │  │  Prometheus     │  │  OpenTelemetry       │   │
│  │  Logging     │  │  Metrics        │  │  Tracing (optional)  │   │
│  │  (structlog) │  │  (/metrics)     │  │                      │   │
│  └──────────────┘  └─────────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Ingestion**: `Document → Load → Chunk (token-aware) → Embed (BGE) → Index (Qdrant)`
2. **Query**: `Question → Embed → Retrieve (2x over-fetch) → Rerank (Cross-Encoder) → Generate (DeepSeek) → Cite`
3. **Streaming**: Same as Query, but generation tokens are streamed via SSE or WebSocket

---

## Key Features

### Production-Ready
- **Error containment**: Every component catches, logs, and surfaces errors through a unified `CognitaError` hierarchy
- **Retry with exponential backoff**: LLM calls automatically retry on transient failures (connection, timeout, rate limit)
- **Graceful degradation**: Reranker, query expander, and HyDE all fall back to no-op behavior when their dependencies are unavailable
- **Non-root Docker user**: Production container runs as an unprivileged `cognita` user
- **Health checks**: Separate liveness (`/health`) and readiness (`/ready`) probes

### Retrieval Quality
- **Hybrid retrieval**: Over-fetches 2x candidates from the vector store, giving the reranker room to optimize ordering
- **Cross-encoder reranking**: Uses `BAAI/bge-reranker-base` to jointly encode (query, document) pairs for precise relevance scoring
- **Query expansion**: LLM-generated alternative phrasings improve recall for queries with unusual vocabulary
- **HyDE**: Generates hypothetical answer documents to bridge the vocabulary gap between questions and answers
- **Conversation context**: Recent dialogue turns are folded into the query embedding for multi-turn disambiguation

### Generation Quality
- **Citation enforcement**: System prompt mandates inline `[1]`, `[2]` citations referencing the provided context
- **Grounding guardrails**: LLM is instructed to answer ONLY from retrieved context; honest refusal when context is insufficient
- **Language matching**: Automatically detects and responds in the user's language
- **Thinking mode**: Optional `deepseek-v4-pro` model for complex multi-step reasoning tasks
- **Token streaming**: Both SSE and WebSocket streaming for real-time response delivery

### Observability
- **Structured logging**: JSON-formatted logs in production, colored console output in development (via `structlog`)
- **Sensitive data redaction**: API keys, tokens, and passwords are automatically scrubbed from log entries
- **Prometheus metrics**: 15+ metrics covering API latency, LLM token usage, embedding duration, retrieval scores, ingestion throughput, and vector store operations
- **OpenTelemetry tracing**: Optional distributed tracing via OTLP exporter

### Security
- **API key authentication**: Supports both `X-API-Key` header and `Authorization: Bearer` token
- **Rate limiting**: Sliding-window per-IP rate limiter (configurable requests/window)
- **CORS**: Configurable allowed origins

---

## Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **LLM** | DeepSeek v4 (flash/pro) | Chat completion + reasoning |
| **Embedding** | BAAI/bge-small-zh-v1.5 | Local sentence embeddings (512-dim) |
| **Reranker** | BAAI/bge-reranker-base | Cross-encoder relevance scoring |
| **Vector DB** | Qdrant | Similarity search + metadata filtering |
| **API** | FastAPI + Uvicorn | REST + WebSocket + SSE |
| **CLI** | Click + Rich | Interactive terminal interface |
| **Logging** | structlog | Structured JSON logging |
| **Metrics** | prometheus-client | Prometheus-compatible metrics |
| **Container** | Docker + Docker Compose | Production deployment |
| **CI/CD** | GitHub Actions | Automated testing + Docker builds |
| **Testing** | pytest + pytest-asyncio | Unit + integration tests |
| **Linting** | Ruff + MyPy | Code quality + type checking |

---

## Project Structure

```
cognita-rag/
├── cognita/                    # Main application package
│   ├── api/                    # FastAPI REST + WebSocket layer
│   │   ├── app.py              # Application factory
│   │   ├── auth.py             # API key authentication
│   │   ├── middleware.py       # Rate limiting + request logging
│   │   ├── routes.py           # All endpoint handlers
│   │   └── schemas.py          # Pydantic request/response models
│   ├── cli/                    # Command-line interface
│   │   └── main.py             # Click + Rich CLI
│   ├── core/                   # Core abstractions
│   │   ├── embedding.py        # Local embedding (sentence-transformers)
│   │   ├── exceptions.py       # Unified error hierarchy
│   │   ├── llm.py              # DeepSeek LLM with retry logic
│   │   ├── models.py           # Domain data models
│   │   └── vectorstore.py      # Qdrant + in-memory vector stores
│   ├── generation/             # Answer generation
│   │   ├── generator.py        # RAG generator (sync + streaming)
│   │   ├── memory.py           # Conversation memory (thread-safe)
│   │   └── prompts.py          # Prompt engineering with citations
│   ├── ingestion/              # Document processing pipeline
│   │   ├── chunkers.py         # Token-aware recursive chunker
│   │   ├── loaders.py          # PDF/MD/TXT/DOCX loaders
│   │   └── pipeline.py         # Orchestrator: load→chunk→embed→index
│   ├── observability/          # Logging + metrics
│   │   ├── logging.py          # structlog configuration
│   │   └── metrics.py          # Prometheus metrics definitions
│   ├── retrieval/              # Retrieval engine
│   │   ├── expander.py         # Query expansion + HyDE
│   │   ├── hybrid.py           # Hybrid retriever (2x over-fetch)
│   │   └── reranker.py         # Cross-encoder reranker
│   └── config.py               # Pydantic settings management
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests (no external deps)
│   └── integration/            # Integration tests (requires Qdrant)
├── documents/                  # Example documents for ingestion
├── docker/                     # Docker configuration
│   └── prometheus.yml          # Prometheus scrape config
├── .github/workflows/          # CI/CD pipeline
│   └── ci.yml                  # Lint → Test → Docker Build → Release
├── Dockerfile                  # Multi-stage production image
├── docker-compose.yml          # Qdrant + API + Prometheus
├── Makefile                    # Development commands
├── pyproject.toml              # Project metadata + dependencies
└── .env.example                # Configuration template
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Docker + Docker Compose (for containerized deployment)
- DeepSeek API key ([get one here](https://platform.deepseek.com/))

### Option 1: Docker Compose (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/ZeffyTheCoder/cognita-rag.git
cd cognita-rag

# 2. Copy and configure environment
cp .env.example .env
# Edit .env and set DEEPSEEK_API_KEY

# 3. Start all services (Qdrant + API)
docker-compose up -d

# 4. Verify the service is healthy
curl http://localhost:8000/health

# 5. Ingest example documents
curl -X POST http://localhost:8000/api/v1/documents/directory \
  -H "Content-Type: application/json" \
  -d '{"directory": "/app/documents"}'

# 6. Ask a question
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG architecture?"}'
```

### Option 2: Local Development

```bash
# 1. Clone and install
git clone https://github.com/ZeffyTheCoder/cognita-rag.git
cd cognita-rag
pip install -e ".[dev]"

# 2. Configure environment
cp .env.example .env
# Edit .env and set DEEPSEEK_API_KEY

# 3. Start Qdrant (via Docker)
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest

# 4. Initialize the system
make init

# 5. Start the API server
make serve

# 6. In a new terminal, ingest documents
cognita ingest documents/

# 7. Query the knowledge base
cognita query "What is RAG architecture?" --show-sources

# 8. Or start an interactive chat
cognita chat
```

---

## Configuration

All configuration is managed through environment variables (or a `.env` file). See `.env.example` for the full template.

### Key Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `DEEPSEEK_API_KEY` | *(required)* | DeepSeek API key |
| `DEEPSEEK_CHAT_MODEL` | `deepseek-v4-flash` | Model for general queries (¥1/M input, ¥2/M output) |
| `DEEPSEEK_REASONING_MODEL` | `deepseek-v4-pro` | Model for thinking mode (¥3/M input, ¥6/M output) |
| `EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | Local embedding model (free, runs on CPU) |
| `EMBEDDING_DEVICE` | `cpu` | Device: `cpu`, `cuda`, or `mps` |
| `VECTOR_STORE_TYPE` | `qdrant` | Vector store: `qdrant` or `memory` |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `API_PORT` | `8000` | API server port |
| `API_KEY` | *(empty)* | Set to enable API key authentication |
| `RATE_LIMIT_REQUESTS` | `60` | Max requests per window per IP |
| `RATE_LIMIT_WINDOW` | `60` | Rate limit window in seconds |
| `LOG_LEVEL` | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `ENVIRONMENT` | `production` | `development` for colored logs, `production` for JSON |
| `CHUNK_SIZE` | `512` | Chunk size in tokens |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks in tokens |
| `RETRIEVAL_TOP_K` | `5` | Number of documents to retrieve |
| `RERANK_ENABLED` | `true` | Enable cross-encoder reranking |
| `ENABLE_THINKING` | `false` | Use reasoning model by default |

### DeepSeek Model Pricing (per 1M tokens)

| Model | Input | Cache Hit | Output | Best For |
|-------|-------|-----------|--------|----------|
| `deepseek-v4-flash` | ¥1 | ¥0.02 | ¥2 | General Q&A, high-volume |
| `deepseek-v4-pro` | ¥3 | ¥0.025 | ¥6 | Complex reasoning, agents |

---

## API Reference

### Health & Readiness

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe (always 200) |
| `GET` | `/ready` | Readiness probe (503 if unhealthy) |
| `GET` | `/metrics` | Prometheus metrics endpoint |

### Document Management

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/documents/text` | Ingest raw text |
| `POST` | `/api/v1/documents` | Upload and ingest a file (multipart) |
| `POST` | `/api/v1/documents/directory` | Ingest all files in a directory |
| `GET` | `/api/v1/documents` | Get document statistics |
| `DELETE` | `/api/v1/documents/{id}` | Delete a document and all chunks |

### Query

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/query` | Query and get a grounded answer with citations |
| `POST` | `/api/v1/query/stream` | Stream answer via Server-Sent Events (SSE) |
| `WS` | `/api/v1/chat` | WebSocket for interactive streaming chat |

### Example: Ingest Text

```bash
curl -X POST http://localhost:8000/api/v1/documents/text \
  -H "Content-Type: application/json" \
  -d '{
    "title": "RAG Overview",
    "content": "Retrieval-Augmented Generation combines retrieval and generation..."
  }'
```

### Example: Query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is RAG?",
    "top_k": 5,
    "thinking": false,
    "conversation_id": "session-1"
  }'
```

**Response:**
```json
{
  "answer": "RAG (Retrieval-Augmented Generation) is a technique that combines retrieval and generation [1]...",
  "citations": [
    {
      "chunk_id": "abc123",
      "document_id": "def456",
      "document_title": "RAG Overview",
      "source": "user_input",
      "content_snippet": "Retrieval-Augmented Generation combines...",
      "score": 0.89,
      "chunk_index": 0
    }
  ],
  "usage": {"prompt_tokens": 512, "completion_tokens": 128, "total_tokens": 640},
  "model": "deepseek-v4-flash",
  "latency_ms": 1234.56
}
```

### Example: WebSocket Chat

```javascript
const ws = new WebSocket("ws://localhost:8000/api/v1/chat");

ws.onopen = () => {
  ws.send(JSON.stringify({
    query: "Explain the ingestion pipeline",
    conversation_id: "session-1",
    thinking: true
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === "token") {
    process.stdout.write(data.token);
  } else if (data.type === "done") {
    console.log("\n\nCitations:", data.citations);
  }
};
```

### Authentication

When `API_KEY` is set in the environment, all endpoints require authentication via either:

```bash
# Option 1: X-API-Key header
curl -H "X-API-Key: your-secret-key" http://localhost:8000/api/v1/query ...

# Option 2: Bearer token
curl -H "Authorization: Bearer your-secret-key" http://localhost:8000/api/v1/query ...
```

---

## CLI Usage

The CLI provides a rich terminal interface powered by Click and Rich.

```bash
# Initialize the system (create vectorstore collection)
cognita init

# Ingest a file or directory
cognita ingest documents/
cognita ingest path/to/document.pdf

# Ask a question
cognita query "What is RAG architecture?" --show-sources

# Ask with thinking mode (uses deepseek-v4-pro)
cognita query "Analyze the trade-offs between bi-encoder and cross-encoder" --thinking

# Start an interactive chat session
cognita chat

# Check system health
cognita health

# List document statistics
cognita list

# Delete a document
cognita delete <document_id>

# Start the API server
cognita serve --host 0.0.0.0 --port 8000 --reload
```

---

## Testing

```bash
# Run all tests
make test

# Run unit tests only (no external dependencies)
make test-unit

# Run integration tests (requires Qdrant)
make test-integration

# Run with coverage
pytest tests/unit/ -v --cov=cognita --cov-report=term --cov-report=html
```

### Test Structure

- **Unit tests** (`tests/unit/`): Test individual components in isolation using the in-memory vector store and mocked LLM. No external services required.
- **Integration tests** (`tests/integration/`): Test the full API stack including FastAPI endpoints with a real Qdrant instance.

---

## Docker Deployment

### Build and Run

```bash
# Build the production image
docker build -t cognita-rag:latest .

# Start all services
docker-compose up -d

# Start with monitoring stack (Prometheus)
docker-compose --profile monitoring up -d

# View logs
docker-compose logs -f cognita

# Stop all services
docker-compose down
```

### Docker Image Features

- **Multi-stage build**: Separate builder and production stages for smaller image size
- **Non-root user**: Runs as `cognita` user for security
- **Health check**: Built-in health check via `/health` endpoint
- **Model caching**: Embedding model is cached in a volume for faster restarts

---

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push and pull request:

1. **Lint & Type Check**: Ruff linter + formatter check + MyPy type checking
2. **Unit Tests**: Run on Python 3.10, 3.11, and 3.12 with coverage reporting
3. **Integration Tests**: Run against a live Qdrant container
4. **Docker Build**: Build the production image (on main/master pushes)
5. **Release**: On GitHub release, push the Docker image to GitHub Container Registry (GHCR)

```yaml
# Triggered on: push to main, pull requests, and releases
# Matrix: Python 3.10, 3.11, 3.12
# Services: Qdrant container for integration tests
# Artifacts: Coverage report, Docker image
```

---

## Observability

### Structured Logging

In production, logs are emitted as JSON for easy ingestion by log aggregation systems:

```json
{
  "event": "LLM chat completed",
  "model": "deepseek-v4-flash",
  "tokens": {"prompt_tokens": 512, "completion_tokens": 128, "total_tokens": 640},
  "latency_ms": 1234.56,
  "app": "Cognita RAG",
  "env": "production",
  "version": "1.0.0",
  "timestamp": "2026-01-15T10:30:00Z",
  "level": "info",
  "logger": "cognita.llm.deepseek"
}
```

### Prometheus Metrics

Access metrics at `http://localhost:8000/metrics`. Key metrics include:

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `cognita_api_requests_total` | Counter | method, endpoint, status | Total API requests |
| `cognita_api_request_duration_seconds` | Histogram | method, endpoint | API request latency |
| `cognita_llm_requests_total` | Counter | model, status | LLM API requests |
| `cognita_llm_tokens_total` | Counter | model, type | Token usage (prompt/completion) |
| `cognita_embedding_requests_total` | Counter | status | Embedding requests |
| `cognita_retrieval_duration_seconds` | Histogram | - | Retrieval latency |
| `cognita_retrieval_score` | Histogram | - | Similarity score distribution |
| `cognita_ingestion_duration_seconds` | Histogram | file_type | Ingestion latency |
| `cognita_vectorstore_collection_size` | Gauge | - | Total vectors in collection |
| `cognita_active_websocket_connections` | Gauge | - | Active WebSocket connections |
| `cognita_errors_total` | Counter | type, severity | Application errors |

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Install dev dependencies: `pip install -e ".[dev]"`
4. Make your changes
5. Run tests: `make test`
6. Run linting: `make lint && make format`
7. Commit with conventional commits
8. Push and create a pull request

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
