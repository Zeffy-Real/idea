# Python Production Best Practices

## Project Structure

A well-organized Python project follows these conventions:

```
project/
├── pyproject.toml          # Project metadata and dependencies
├── README.md               # Project documentation
├── LICENSE                 # License file
├── .env.example            # Environment variable template
├── .gitignore              # Git ignore rules
├── src/                    # Source code
│   └── package_name/
│       ├── __init__.py
│       ├── core/           # Core abstractions
│       ├── api/            # API layer
│       └── utils/          # Utilities
├── tests/                  # Test suite
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── conftest.py         # Pytest fixtures
├── docker/                 # Docker configurations
└── .github/workflows/      # CI/CD pipelines
```

## Dependency Management

- Use `pyproject.toml` for all project metadata
- Pin major versions with `>=` for compatibility
- Separate dev dependencies using optional-dependencies
- Use `pip install -e ".[dev]"` for development

## Error Handling

### Exception Hierarchy

Design a custom exception hierarchy that maps to your domain:

```python
class AppError(Exception):
    """Base exception with structured details."""
    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

class ConfigError(AppError): ...
class LLMError(AppError): ...
class DatabaseError(AppError): ...
```

### Retry Logic

Use tenacity for exponential backoff retries:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
async def call_external_api():
    ...
```

## Async/Await Patterns

- Use `asyncio` for I/O-bound operations
- Use `asyncio.gather()` for parallel operations
- Offload CPU-bound work with `loop.run_in_executor()`
- Always properly close resources with `async with` or cleanup methods

## Configuration Management

- Use `pydantic-settings` for typed configuration
- Load from `.env` files and environment variables
- Cache settings with `@lru_cache`
- Validate all configuration at startup

## Logging

- Use structured logging (structlog) for machine-readable logs
- Include request IDs, timestamps, and contextual metadata
- Redact sensitive information (API keys, passwords)
- Use appropriate log levels (DEBUG, INFO, WARNING, ERROR)

## Testing

### Unit Tests

- Test individual functions and classes in isolation
- Mock external dependencies
- Use fixtures for setup and teardown
- Aim for high coverage of critical paths

### Integration Tests

- Test component interactions
- Use real databases (or test containers)
- Test error scenarios and edge cases
- Mark slow tests with `@pytest.mark.slow`

### Test Configuration

```python
# conftest.py
import pytest

@pytest.fixture
def settings():
    """Override settings for testing."""
    return Settings(env_file=None, deepseek_api_key="test-key")

@pytest.fixture
async def vectorstore():
    """In-memory vector store for testing."""
    store = InMemoryVectorStore(dimension=512)
    await store.create_collection(512)
    return store
```

## Docker Best Practices

- Use multi-stage builds to reduce image size
- Run as non-root user
- Use `.dockerignore` to exclude unnecessary files
- Set health checks for container orchestration
- Use `PYTHONUNBUFFERED=1` for real-time logs

## CI/CD Pipeline

- Run linting (ruff) and type checking (mypy) on every push
- Run tests on multiple Python versions
- Build Docker image on main branch
- Publish images on releases
- Use caching for dependencies

## Security

- Never commit secrets to version control
- Use environment variables for sensitive configuration
- Validate all user input
- Implement rate limiting for APIs
- Use HTTPS for all external communication
- Keep dependencies updated
