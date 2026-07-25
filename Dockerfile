# =============================================================================
# Cognita RAG - Dockerfile (Multi-stage for optimized production image)
# =============================================================================

FROM python:3.10-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only dependency files for better caching
COPY pyproject.toml ./

# Install dependencies to a separate directory
RUN pip install --no-cache-dir --prefix=/install \
    pydantic>=2.5.0 \
    pydantic-settings>=2.1.0 \
    python-dotenv>=1.0.0 \
    openai>=1.12.0 \
    httpx>=0.27.0 \
    tenacity>=8.2.3 \
    sentence-transformers>=2.5.0 \
    numpy>=1.24.0 \
    qdrant-client>=1.8.0 \
    pypdf>=4.0.0 \
    python-docx>=1.1.0 \
    markdown>=3.5.0 \
    beautifulsoup4>=4.12.0 \
    tiktoken>=0.6.0 \
    fastapi>=0.110.0 \
    "uvicorn[standard]>=0.29.0" \
    websockets>=12.0 \
    python-multipart>=0.0.9 \
    slowapi>=0.1.9 \
    structlog>=24.1.0 \
    prometheus-client>=0.20.0 \
    click>=8.1.7 \
    rich>=13.7.0

# =============================================================================
# Production stage
# =============================================================================

FROM python:3.10-slim AS production

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r cognita && useradd -r -g cognita -s /bin/bash cognita

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=cognita:cognita . /app/

# Create data directory for models
RUN mkdir -p /app/data/models && chown -R cognita:cognita /app

# Switch to non-root user
USER cognita

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    EMBEDDING_DEVICE=cpu \
    ENVIRONMENT=production \
    LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Expose API port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "uvicorn", "cognita.api.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--access-log"]
