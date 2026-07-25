"""
Structured logging configuration using structlog.
Provides JSON-formatted logs in production, human-readable in development.
"""

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, Processor

from cognita.config import get_settings


def _add_app_context(_logger: Any, _method_name: str, event_dict: EventDict) -> EventDict:
    """Add application context to every log entry."""
    settings = get_settings()
    event_dict["app"] = settings.app_name
    event_dict["env"] = settings.environment
    event_dict["version"] = settings.app_version
    return event_dict


def _add_logger_name(logger: Any, _method_name: str, event_dict: EventDict) -> EventDict:
    """Add the logger name to the event dict.

    A drop-in replacement for :func:`structlog.stdlib.add_logger_name` that
    gracefully handles loggers without a ``name`` attribute (e.g. the
    ``PrintLogger`` used by :class:`structlog.PrintLoggerFactory`).
    """
    record = event_dict.get("_record")
    if record is not None:
        event_dict["logger"] = record.name
    else:
        name = getattr(logger, "name", None)
        if name is not None:
            event_dict["logger"] = name
    return event_dict


def _filter_sensitive_keys(
    _logger: Any, _method_name: str, event_dict: EventDict
) -> EventDict:
    """Redact sensitive values from log entries."""
    sensitive_keys = {"api_key", "password", "token", "secret", "authorization"}
    for key in list(event_dict.keys()):
        lower_key = key.lower()
        if any(s in lower_key for s in sensitive_keys):
            event_dict[key] = "[REDACTED]"
    return event_dict


def setup_logging() -> None:
    """Configure structlog and standard logging integration."""
    settings = get_settings()

    # Shared processors for both structlog and stdlib
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        _add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        _add_app_context,
        _filter_sensitive_keys,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # Configure structlog
    if settings.environment == "development":
        # Human-readable console output for development
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        # JSON output for production (log aggregation)
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    # Configure stdlib logging to route through structlog
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, settings.log_level.upper()))

    # Reduce noise from third-party libraries
    for noisy in ["httpx", "httpcore", "openai", "urllib3", "sentence_transformers"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)
