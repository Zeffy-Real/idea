"""
Custom exception hierarchy for the application.
Provides granular error types for different failure modes.
"""


class CognitaError(Exception):
    """Base exception for all application errors."""

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        return self.message


class ConfigError(CognitaError):
    """Configuration-related errors."""


class LLMError(CognitaError):
    """LLM provider errors."""

    def __init__(self, message: str, *, provider: str = "", status_code: int = 0, **kwargs):
        super().__init__(message, **kwargs)
        self.provider = provider
        self.status_code = status_code


class LLMRateLimitError(LLMError):
    """LLM rate limit exceeded."""

    def __init__(self, message: str, *, retry_after: float = 0.0, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class LLMTimeoutError(LLMError):
    """LLM request timed out."""


class EmbeddingError(CognitaError):
    """Embedding model errors."""

    def __init__(self, message: str, *, model: str = "", **kwargs):
        super().__init__(message, **kwargs)
        self.model = model


class VectorStoreError(CognitaError):
    """Vector store operation errors."""

    def __init__(self, message: str, *, store: str = "", operation: str = "", **kwargs):
        super().__init__(message, **kwargs)
        self.store = store
        self.operation = operation


class VectorStoreConnectionError(VectorStoreError):
    """Cannot connect to vector store."""


class DocumentLoadingError(CognitaError):
    """Document loading/parsing errors."""

    def __init__(self, message: str, *, file_path: str = "", **kwargs):
        super().__init__(message, **kwargs)
        self.file_path = file_path


class ChunkingError(CognitaError):
    """Text chunking errors."""


class RetrievalError(CognitaError):
    """Document retrieval errors."""


class GenerationError(CognitaError):
    """Answer generation errors."""


class AuthenticationError(CognitaError):
    """Authentication failures."""

    def __init__(self, message: str = "Authentication required", **kwargs):
        super().__init__(message, **kwargs)


class RateLimitExceededError(CognitaError):
    """Rate limit exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", **kwargs):
        super().__init__(message, **kwargs)
