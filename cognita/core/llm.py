"""
LLM abstraction layer with DeepSeek implementation.
Provides a unified interface for chat completion, streaming, and thinking mode.
Uses the OpenAI-compatible API with retry logic and proper error handling.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from cognita.config import Settings, get_settings
from cognita.core.exceptions import (
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from cognita.core.models import Message
from cognita.observability.logging import get_logger
from cognita.observability.metrics import (
    llm_request_duration,
    llm_requests_total,
    llm_tokens_used,
)

logger = get_logger(__name__)


class BaseLLM(ABC):
    """Abstract LLM interface for chat completion."""

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        thinking: bool = False,
        **kwargs: Any,
    ) -> tuple[str, dict[str, int], str | None]:
        """Generate a chat completion.

        Returns:
            Tuple of (content, usage_dict, thinking_content).
        """

    @abstractmethod
    async def chat_stream(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        thinking: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a chat completion token by token."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the LLM provider is reachable."""


class DeepSeekLLM(BaseLLM):
    """DeepSeek LLM implementation using OpenAI-compatible API."""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._client: AsyncOpenAI | None = None
        self._logger = get_logger("cognita.llm.deepseek")

    @property
    def client(self) -> AsyncOpenAI:
        """Lazy-initialize the AsyncOpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._settings.deepseek_api_key,
                base_url=self._settings.deepseek_base_url,
                timeout=httpx.Timeout(60.0, connect=10.0),
                max_retries=0,  # We handle retries ourselves
            )
        return self._client

    def _get_model(self, thinking: bool) -> str:
        """Select model based on thinking mode."""
        if thinking:
            return self._settings.deepseek_reasoning_model
        return self._settings.deepseek_chat_model

    def _build_messages(
        self, messages: list[Message]
    ) -> list[dict[str, str]]:
        """Convert Message objects to API format."""
        return [{"role": m.role, "content": m.content} for m in messages]

    @retry(
        retry=retry_if_exception_type(
            (APIConnectionError, APITimeoutError, LLMRateLimitError)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        thinking: bool = False,
        **kwargs: Any,
    ) -> tuple[str, dict[str, int], str | None]:
        """Generate a chat completion via DeepSeek API."""
        model = self._get_model(thinking)
        api_messages = self._build_messages(messages)

        self._logger.debug(
            "LLM chat request",
            model=model,
            message_count=len(api_messages),
            thinking=thinking,
        )

        start = time.perf_counter()
        status = "success"

        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=api_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                **kwargs,
            )

            content = response.choices[0].message.content or ""
            usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }

            # Extract thinking content if present (DeepSeek reasoning model)
            thinking_content = None
            if thinking and hasattr(response.choices[0].message, "reasoning_content"):
                thinking_content = getattr(
                    response.choices[0].message, "reasoning_content", None
                )

            # Record metrics
            elapsed = time.perf_counter() - start
            llm_requests_total.labels(model=model, status=status).inc()
            llm_request_duration.labels(model=model).observe(elapsed)
            llm_tokens_used.labels(model=model, type="prompt").inc(usage["prompt_tokens"])
            llm_tokens_used.labels(model=model, type="completion").inc(
                usage["completion_tokens"]
            )

            self._logger.info(
                "LLM chat completed",
                model=model,
                tokens=usage,
                latency_ms=round(elapsed * 1000, 2),
            )

            return content, usage, thinking_content

        except RateLimitError as e:
            status = "rate_limited"
            retry_after = getattr(e, "retry_after", 0) or 0
            self._logger.warning("LLM rate limited", model=model, retry_after=retry_after)
            raise LLMRateLimitError(
                f"Rate limit exceeded: {e}",
                provider="deepseek",
                retry_after=retry_after,
            ) from e

        except APITimeoutError as e:
            status = "timeout"
            self._logger.error("LLM request timed out", model=model)
            raise LLMTimeoutError(
                f"Request timed out: {e}", provider="deepseek"
            ) from e

        except APIConnectionError as e:
            status = "connection_error"
            self._logger.error("LLM connection error", model=model, error=str(e))
            raise LLMError(
                f"Connection error: {e}", provider="deepseek"
            ) from e

        except APIStatusError as e:
            status = f"error_{e.status_code}"
            self._logger.error(
                "LLM API error",
                model=model,
                status_code=e.status_code,
                error=str(e),
            )
            raise LLMError(
                f"API error ({e.status_code}): {e.message}",
                provider="deepseek",
                status_code=e.status_code,
            ) from e

        finally:
            if status != "success":
                llm_requests_total.labels(model=model, status=status).inc()

    @retry(
        retry=retry_if_exception_type(
            (APIConnectionError, APITimeoutError, LLMRateLimitError)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def chat_stream(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        thinking: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a chat completion token by token."""
        model = self._get_model(thinking)
        api_messages = self._build_messages(messages)

        self._logger.debug(
            "LLM stream request",
            model=model,
            message_count=len(api_messages),
            thinking=thinking,
        )

        start = time.perf_counter()
        total_tokens = 0

        try:
            stream = await self.client.chat.completions.create(
                model=model,
                messages=api_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                stream_options={"include_usage": True},
                **kwargs,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    total_tokens += 1
                    yield chunk.choices[0].delta.content

                # Capture usage from the final chunk
                if chunk.usage:
                    llm_tokens_used.labels(model=model, type="prompt").inc(
                        chunk.usage.prompt_tokens
                    )
                    llm_tokens_used.labels(model=model, type="completion").inc(
                        chunk.usage.completion_tokens
                    )

            elapsed = time.perf_counter() - start
            llm_requests_total.labels(model=model, status="success").inc()
            llm_request_duration.labels(model=model).observe(elapsed)

            self._logger.info(
                "LLM stream completed",
                model=model,
                chunks_yielded=total_tokens,
                latency_ms=round(elapsed * 1000, 2),
            )

        except RateLimitError as e:
            llm_requests_total.labels(model=model, status="rate_limited").inc()
            raise LLMRateLimitError(
                f"Rate limit exceeded: {e}", provider="deepseek"
            ) from e
        except APITimeoutError as e:
            llm_requests_total.labels(model=model, status="timeout").inc()
            raise LLMTimeoutError(
                f"Request timed out: {e}", provider="deepseek"
            ) from e
        except APIConnectionError as e:
            llm_requests_total.labels(model=model, status="connection_error").inc()
            raise LLMError(
                f"Connection error: {e}", provider="deepseek"
            ) from e
        except APIStatusError as e:
            llm_requests_total.labels(model=model, status=f"error_{e.status_code}").inc()
            raise LLMError(
                f"API error ({e.status_code}): {e.message}",
                provider="deepseek",
                status_code=e.status_code,
            ) from e

    async def health_check(self) -> bool:
        """Check if the DeepSeek API is reachable."""
        try:
            models = await self.client.models.list()
            return len(models.data) > 0
        except Exception as e:
            self._logger.warning("Health check failed", error=str(e))
            return False

    async def close(self) -> None:
        """Clean up resources."""
        if self._client:
            await self._client.close()
            self._client = None


# Factory function for dependency injection
_llm_instance: DeepSeekLLM | None = None


def get_llm() -> DeepSeekLLM:
    """Get the singleton LLM instance."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = DeepSeekLLM()
    return _llm_instance


async def reset_llm() -> None:
    """Reset the LLM instance (useful for testing)."""
    global _llm_instance
    if _llm_instance is not None:
        await _llm_instance.close()
    _llm_instance = None
