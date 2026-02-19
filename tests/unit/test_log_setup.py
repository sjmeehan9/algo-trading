"""Tests for logger setup configuration."""

from __future__ import annotations

import json
import logging

from algotrading.src.log_setup import JsonLogFormatter, setup_logger


def test_setup_logger_returns_logger(tmp_path) -> None:
    """setup_logger returns a configured logger and attaches handlers."""

    logger = setup_logger(str(tmp_path), log_level="INFO", print_logs=False)

    assert isinstance(logger, logging.Logger)
    assert len(logging.getLogger().handlers) >= 1


def test_setup_logger_json_formatter(tmp_path) -> None:
    """setup_logger supports JSON log formatting."""

    setup_logger(str(tmp_path), log_level="DEBUG", print_logs=False, use_json=True)
    formatter = logging.getLogger().handlers[0].formatter

    assert isinstance(formatter, JsonLogFormatter)

    record = logging.LogRecord(
        name="algotrading.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=12,
        msg="hello",
        args=(),
        exc_info=None,
    )
    payload = json.loads(formatter.format(record))
    assert payload["level"] == "INFO"
    assert payload["message"] == "hello"
    assert payload["module"] == "algotrading.test"


def test_setup_logger_invalid_level_raises(tmp_path) -> None:
    """Invalid log level strings should raise ValueError."""

    try:
        setup_logger(str(tmp_path), log_level="NOT_A_LEVEL")
    except ValueError as exc:
        assert "Invalid log level" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid log level")
