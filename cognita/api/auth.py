"""API key authentication for the FastAPI application.

Supports two methods of supplying the API key:

  1. ``X-API-Key`` header (preferred).
  2. ``Authorization: Bearer <key>`` header (standard OAuth2 pattern).

When ``settings.api_key`` is empty, authentication is disabled entirely and
all requests are allowed through.
"""

from __future__ import annotations

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from cognita.config import get_settings
from cognita.observability.logging import get_logger

logger = get_logger("cognita.api.auth")

# Header-based API key extractor (does not auto-error so we can provide a
# custom 401 response with helpful details).
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Bearer-token extractor (falls back when the X-API-Key header is absent).
_bearer_scheme = HTTPBearer(auto_error=False)


def verify_api_key(api_key: str | None) -> bool:
    """Verify *api_key* against the configured key.

    Returns ``True`` when:

      * Authentication is disabled (``settings.api_key`` is empty), **or**
      * *api_key* matches the configured key.

    Returns ``False`` otherwise.
    """
    settings = get_settings()

    # Auth disabled — allow all requests.
    if not settings.api_key:
        return True

    if api_key is None:
        return False

    # Constant-time-ish comparison (not a hard security requirement here since
    # the key is a shared secret, but avoids trivial timing oracles).
    return api_key == settings.api_key


async def APIKeyDependency(
    api_key_header: str | None = Security(_api_key_header),
    bearer: HTTPAuthorizationCredentials | None = Security(_bearer_scheme),
) -> bool:
    """FastAPI dependency that extracts and verifies the API key.

    The key is read from the ``X-API-Key`` header first, falling back to the
    ``Authorization: Bearer <key>`` header.  When authentication is enabled
    (``settings.api_key`` is non-empty) and the key is missing or invalid, an
    ``HTTP 401`` is raised.
    """
    settings = get_settings()

    # Authentication disabled — short-circuit.
    if not settings.auth_enabled:
        return True

    # Prefer the dedicated header; fall back to the bearer token.
    extracted_key: str | None = api_key_header
    if extracted_key is None and bearer is not None:
        extracted_key = bearer.credentials

    if not verify_api_key(extracted_key):
        logger.warning(
            "Authentication failed",
            has_header=api_key_header is not None,
            has_bearer=bearer is not None,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Provide it via the 'X-API-Key' "
            "header or the 'Authorization: Bearer <key>' header.",
            headers={"WWW-Authenticate": 'Bearer realm="cognita"'},
        )

    return True
