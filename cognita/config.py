"""
Centralized configuration management with environment variable support.
Uses pydantic-settings for typed, validated configuration.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # DeepSeek API
    deepseek_api_key: str = Field(..., description="DeepSeek API key")
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com",
        description="DeepSeek API base URL",
    )
    deepseek_chat_model: str = Field(
        default="deepseek-v4-flash",
        description="DeepSeek chat model (non-thinking mode)",
    )
    deepseek_reasoning_model: str = Field(
        default="deepseek-v4-pro",
        description="DeepSeek reasoning model (thinking mode)",
    )

    # Embedding
    embedding_model: str = Field(
        default="BAAI/bge-small-zh-v1.5",
        description="Sentence-transformers model name",
    )
    embedding_dimension: int = Field(default=512, description="Embedding vector dimension")
    embedding_device: str = Field(default="cpu", description="Device for embedding model")

    # Vector Store
    vector_store_type: Literal["qdrant", "memory"] = Field(
        default="qdrant",
        description="Vector store backend type",
    )
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant server URL")
    qdrant_collection: str = Field(
        default="cognita_docs",
        description="Qdrant collection name",
    )
    qdrant_api_key: str = Field(default="", description="Qdrant API key (optional)")

    # API Server
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")
    api_key: str = Field(default="", description="API key for authentication (empty = no auth)")
    cors_origins: str = Field(default="*", description="CORS allowed origins")

    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests: int = Field(default=60, description="Max requests per window")
    rate_limit_window: int = Field(default=60, description="Rate limit window in seconds")

    # Observability
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    enable_tracing: bool = Field(default=False, description="Enable OpenTelemetry tracing")
    otel_endpoint: str = Field(
        default="http://localhost:4317",
        description="OpenTelemetry collector endpoint",
    )

    # Application
    app_name: str = Field(default="Cognita RAG", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    environment: Literal["development", "staging", "production"] = Field(
        default="production",
        description="Running environment",
    )

    # Chunking
    chunk_size: int = Field(default=512, description="Default chunk size in tokens")
    chunk_overlap: int = Field(default=64, description="Chunk overlap in tokens")

    # Retrieval
    retrieval_top_k: int = Field(default=5, description="Number of documents to retrieve")
    retrieval_score_threshold: float = Field(
        default=0.3,
        description="Minimum similarity score threshold",
    )
    rerank_enabled: bool = Field(default=True, description="Enable cross-encoder reranking")
    rerank_model: str = Field(
        default="BAAI/bge-reranker-base",
        description="Cross-encoder reranker model",
    )

    # Generation
    max_tokens: int = Field(default=2048, description="Max tokens for LLM response")
    temperature: float = Field(default=0.3, description="LLM temperature for generation")
    enable_thinking: bool = Field(
        default=False,
        description="Use thinking mode for complex queries",
    )
    conversation_memory_turns: int = Field(
        default=10,
        description="Number of conversation turns to keep in memory",
    )

    # Paths
    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    @property
    def documents_dir(self) -> Path:
        return self.project_root / "documents"

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def cors_origins_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @field_validator("api_key")
    @classmethod
    def normalize_api_key(cls, v: str) -> str:
        return v.strip()

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def reload_settings() -> Settings:
    """Force reload settings (useful for testing)."""
    get_settings.cache_clear()
    return get_settings()
