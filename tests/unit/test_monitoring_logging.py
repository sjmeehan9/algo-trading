"""Unit tests for structured monitoring logging helpers."""

from __future__ import annotations

import json
import logging

from algotrading.src.monitoring import (
    StructuredLogFormatter,
    log_event,
    reset_correlation_id,
    set_correlation_id,
)


def test_structured_log_formatter_includes_correlation_and_extra_fields() -> None:
    """Structured logs should include standard fields, extras, and correlation ID."""

    formatter = StructuredLogFormatter()
    token = set_correlation_id("trace-123")
    try:
        record = logging.LogRecord(
            name="algotrading.unit",
            level=logging.INFO,
            pathname=__file__,
            lineno=12,
            msg="hello %s",
            args=("world",),
            exc_info=None,
        )
        record.event_type = "unit.event"
        record.symbol = "AAPL"

        payload = json.loads(formatter.format(record))
    finally:
        reset_correlation_id(token)

    assert payload["level"] == "INFO"
    assert payload["message"] == "hello world"
    assert payload["logger"] == "algotrading.unit"
    assert payload["correlation_id"] == "trace-123"
    assert payload["event_type"] == "unit.event"
    assert payload["symbol"] == "AAPL"


def test_log_event_attaches_event_type(caplog) -> None:
    """log_event should attach structured event metadata to log records."""

    logger = logging.getLogger("algotrading.monitoring.test")

    with caplog.at_level(logging.INFO, logger="algotrading.monitoring.test"):
        log_event(logger, "unit.started", "started", component="monitoring")

    assert caplog.records[0].event_type == "unit.started"
    assert caplog.records[0].component == "monitoring"
