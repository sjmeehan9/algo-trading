"""Structured logging helpers with request correlation context."""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")

_RESERVED_LOG_RECORD_FIELDS: Final[set[str]] = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "message",
    "module",
    "msecs",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


class StructuredLogFormatter(logging.Formatter):
    """Format Python log records as JSON objects with standard fields."""

    def format(self, record: logging.LogRecord) -> str:
        """Return a JSON string for one log record."""

        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.name,
            "source_module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
            "thread": record.threadName,
            "process": record.process,
        }

        corr_id = str(getattr(record, "correlation_id", "") or correlation_id.get())
        if corr_id:
            payload["correlation_id"] = corr_id

        for field_name, field_value in record.__dict__.items():
            if field_name in _RESERVED_LOG_RECORD_FIELDS or field_name.startswith("_"):
                continue
            payload[field_name] = _json_safe(field_value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def set_correlation_id(value: str) -> Token[str]:
    """Set the current request correlation ID and return a reset token."""

    normalized = value.strip()
    return correlation_id.set(normalized)


def reset_correlation_id(token: Token[str]) -> None:
    """Reset the correlation ID context to its previous value."""

    correlation_id.reset(token)


def get_correlation_id() -> str:
    """Return the active request correlation ID, if one is set."""

    return correlation_id.get()


def setup_structured_logging(
    *,
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
    force: bool = False,
) -> None:
    """Configure root logging handlers to emit structured JSON records.

    Args:
        level: Logging level for the root logger and handlers.
        log_file: Optional file path that should also receive JSON logs.
        force: When True, remove existing handlers before installing new ones.
    """

    resolved_level = _coerce_level(level)
    formatter = StructuredLogFormatter()
    root_logger = logging.getLogger()
    root_logger.setLevel(resolved_level)

    if force:
        root_logger.handlers.clear()

    if not root_logger.handlers:
        root_logger.addHandler(logging.StreamHandler())

    for handler in root_logger.handlers:
        handler.setLevel(resolved_level)
        handler.setFormatter(formatter)

    if log_file is not None:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(resolved_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def log_event(
    logger: logging.Logger,
    event_type: str,
    message: str,
    *,
    level: int = logging.INFO,
    exc_info: object = False,
    **fields: object,
) -> None:
    """Log a structured event with consistent event metadata."""

    normalized_event_type = event_type.strip()
    if not normalized_event_type:
        raise ValueError("event_type must be non-empty")

    extra = {"event_type": normalized_event_type}
    extra.update(
        {
            field_name: _json_safe(field_value)
            for field_name, field_value in fields.items()
            if field_name not in _RESERVED_LOG_RECORD_FIELDS
        }
    )
    logger.log(level, message, extra=extra, exc_info=exc_info)


def _coerce_level(level: int | str) -> int:
    if isinstance(level, int):
        return level
    resolved = logging.getLevelName(level.upper())
    if not isinstance(resolved, int):
        raise ValueError(f"Invalid log level: {level}")
    return resolved


def _json_safe(value: object) -> object:
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


__all__ = [
    "StructuredLogFormatter",
    "correlation_id",
    "get_correlation_id",
    "log_event",
    "reset_correlation_id",
    "set_correlation_id",
    "setup_structured_logging",
]
