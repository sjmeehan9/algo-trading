"""Application exception hierarchy for algo-trading."""

from __future__ import annotations

from typing import Any


class AlgoTradingError(Exception):
    """Base exception for all application-specific errors.

    Args:
        message: Human-readable error message.
        context: Optional dictionary of additional debug context.
    """

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def _format_context(self) -> str:
        if not self.context:
            return ""
        context_parts = [
            f"{key}={value!r}" for key, value in sorted(self.context.items())
        ]
        return f" | context: {', '.join(context_parts)}"

    def __str__(self) -> str:
        return f"{self.message}{self._format_context()}"


class ConfigurationError(AlgoTradingError):
    """Raised when configuration loading or validation fails."""

    def __init__(
        self,
        message: str,
        config_file: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        merged_context = dict(context or {})
        if config_file is not None:
            merged_context["config_file"] = config_file
        self.config_file = config_file
        super().__init__(message=message, context=merged_context)


class DataError(AlgoTradingError):
    """Raised when data loading, validation, or processing fails."""

    def __init__(
        self,
        message: str,
        data_source: str | None = None,
        row_count: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        merged_context = dict(context or {})
        if data_source is not None:
            merged_context["data_source"] = data_source
        if row_count is not None:
            merged_context["row_count"] = row_count
        self.data_source = data_source
        self.row_count = row_count
        super().__init__(message=message, context=merged_context)


class BrokerConnectionError(AlgoTradingError):
    """Raised when broker connection or broker API operations fail."""

    def __init__(
        self,
        message: str,
        broker_name: str | None = None,
        host: str | None = None,
        port: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        merged_context = dict(context or {})
        if broker_name is not None:
            merged_context["broker_name"] = broker_name
        if host is not None:
            merged_context["host"] = host
        if port is not None:
            merged_context["port"] = port
        self.broker_name = broker_name
        self.host = host
        self.port = port
        super().__init__(message=message, context=merged_context)


class TrainingError(AlgoTradingError):
    """Raised when model training or evaluation workflows fail."""

    def __init__(
        self,
        message: str,
        model_name: str | None = None,
        step_count: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        merged_context = dict(context or {})
        if model_name is not None:
            merged_context["model_name"] = model_name
        if step_count is not None:
            merged_context["step_count"] = step_count
        self.model_name = model_name
        self.step_count = step_count
        super().__init__(message=message, context=merged_context)


class BacktestError(AlgoTradingError):
    """Raised when backtesting workflows fail."""
