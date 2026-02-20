"""Tests for application custom exceptions."""

from __future__ import annotations

from algotrading.src.exceptions import (
    AlgoTradingError,
    BacktestError,
    BrokerConnectionError,
    ConfigurationError,
    DataError,
    TrainingError,
)


def test_base_exception_formats_context() -> None:
    """Base exception includes sorted context in string output."""

    exc = AlgoTradingError("base failure", context={"b": 2, "a": 1})

    text = str(exc)
    assert "base failure" in text
    assert "a=1" in text
    assert "b=2" in text


def test_configuration_error_includes_config_file() -> None:
    """ConfigurationError should include the config_file context."""

    exc = ConfigurationError("bad config", config_file="config.yml")

    text = str(exc)
    assert "bad config" in text
    assert "config_file='config.yml'" in text


def test_data_error_includes_source_and_rows() -> None:
    """DataError should include data source metadata when provided."""

    exc = DataError("data issue", data_source="historical", row_count=123)

    text = str(exc)
    assert "data issue" in text
    assert "data_source='historical'" in text
    assert "row_count=123" in text


def test_broker_connection_error_includes_endpoint() -> None:
    """BrokerConnectionError includes host and port context."""

    exc = BrokerConnectionError(
        "broker down",
        broker_name="interactive_brokers",
        host="127.0.0.1",
        port=7497,
    )

    text = str(exc)
    assert "broker down" in text
    assert "broker_name='interactive_brokers'" in text
    assert "host='127.0.0.1'" in text
    assert "port=7497" in text


def test_training_error_includes_model_details() -> None:
    """TrainingError includes model metadata in string output."""

    exc = TrainingError("training failed", model_name="ppo", step_count=1000)

    text = str(exc)
    assert "training failed" in text
    assert "model_name='ppo'" in text
    assert "step_count=1000" in text


def test_backtest_error_inherits_base_type() -> None:
    """BacktestError is part of the custom exception hierarchy."""

    exc = BacktestError("backtest failed")

    assert isinstance(exc, AlgoTradingError)
    assert str(exc) == "backtest failed"
