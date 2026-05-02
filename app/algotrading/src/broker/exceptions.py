"""Exceptions for broker adapter operations.

This module defines the canonical exception hierarchy used by broker adapters.
"""

from __future__ import annotations

from dataclasses import dataclass

from algotrading.src.broker.models import ContractSpec, OrderSpec


@dataclass(kw_only=True)
class BrokerError(Exception):
    """Base exception for all broker adapter failures.

    Args:
        message: Human-readable failure message.
        broker_name: Optional broker adapter name.
    """

    message: str
    broker_name: str | None = None

    def __post_init__(self) -> None:
        super().__init__(self.__str__())

    def __str__(self) -> str:
        """Return a readable error representation."""

        if self.broker_name:
            return f"[{self.broker_name}] {self.message}"
        return self.message


@dataclass(kw_only=True)
class BrokerConnectionError(BrokerError):
    """Raised when a broker connection operation fails.

    Args:
        message: Human-readable failure message.
        host: Broker host used for connection.
        port: Broker port used for connection.
        reason: Optional lower-level error reason.
        broker_name: Optional broker adapter name.
    """

    host: str
    port: int
    reason: str | None = None

    def __str__(self) -> str:
        """Return a readable connection error representation."""

        context = f"{self.host}:{self.port}"
        detail = f" reason={self.reason}" if self.reason else ""
        if self.broker_name:
            return f"[{self.broker_name}] {self.message} ({context}){detail}"
        return f"{self.message} ({context}){detail}"


@dataclass(kw_only=True)
class BrokerOrderError(BrokerError):
    """Raised when an order operation fails.

    Args:
        message: Human-readable failure message.
        order_id: Optional broker-assigned order identifier.
        reason: Optional lower-level error reason.
        original_order: Optional original broker-agnostic order specification.
        broker_name: Optional broker adapter name.
    """

    order_id: str | None = None
    reason: str | None = None
    original_order: OrderSpec | None = None

    def __str__(self) -> str:
        """Return a readable order error representation."""

        details: list[str] = []
        if self.order_id:
            details.append(f"order_id={self.order_id}")
        if self.reason:
            details.append(f"reason={self.reason}")
        suffix = f" ({', '.join(details)})" if details else ""
        if self.broker_name:
            return f"[{self.broker_name}] {self.message}{suffix}"
        return f"{self.message}{suffix}"


@dataclass(kw_only=True)
class BrokerDataError(BrokerError):
    """Raised when market/account data retrieval fails.

    Args:
        message: Human-readable failure message.
        contract: Optional broker-agnostic contract requested.
        reason: Optional lower-level error reason.
        broker_name: Optional broker adapter name.
    """

    contract: ContractSpec | None = None
    reason: str | None = None

    def __str__(self) -> str:
        """Return a readable data error representation."""

        details: list[str] = []
        if self.contract:
            details.append(f"symbol={self.contract.symbol}")
        if self.reason:
            details.append(f"reason={self.reason}")
        suffix = f" ({', '.join(details)})" if details else ""
        if self.broker_name:
            return f"[{self.broker_name}] {self.message}{suffix}"
        return f"{self.message}{suffix}"


@dataclass(kw_only=True)
class BrokerTimeoutError(BrokerError):
    """Raised when a broker operation times out.

    Args:
        message: Human-readable failure message.
        operation: Name of the timed-out operation.
        timeout_seconds: Timeout duration in seconds.
        broker_name: Optional broker adapter name.
    """

    operation: str
    timeout_seconds: float

    def __str__(self) -> str:
        """Return a readable timeout error representation."""

        context = f"operation={self.operation}, timeout={self.timeout_seconds}s"
        if self.broker_name:
            return f"[{self.broker_name}] {self.message} ({context})"
        return f"{self.message} ({context})"


@dataclass(kw_only=True)
class BrokerConfigurationError(BrokerError):
    """Raised when broker registry configuration cannot be loaded.

    Args:
        message: Human-readable failure message.
        config_path: Optional path to the broker configuration file.
        reason: Optional lower-level error reason.
        broker_name: Optional broker adapter name.
    """

    config_path: str | None = None
    reason: str | None = None

    def __str__(self) -> str:
        """Return a readable configuration error representation."""

        details: list[str] = []
        if self.config_path:
            details.append(f"config_path={self.config_path}")
        if self.reason:
            details.append(f"reason={self.reason}")
        suffix = f" ({', '.join(details)})" if details else ""
        if self.broker_name:
            return f"[{self.broker_name}] {self.message}{suffix}"
        return f"{self.message}{suffix}"


@dataclass(kw_only=True)
class NoBrokerAvailableError(BrokerError):
    """Raised when no configured broker can be connected.

    Args:
        message: Human-readable failure message.
        attempted_brokers: Broker names attempted in priority order.
        failure_reasons: Per-broker failure summaries safe for logs and UI.
        broker_name: Optional broker adapter name.
    """

    attempted_brokers: tuple[str, ...] = ()
    failure_reasons: tuple[str, ...] = ()

    def __str__(self) -> str:
        """Return a readable no-broker-available error representation."""

        details: list[str] = []
        if self.attempted_brokers:
            details.append(f"attempted={','.join(self.attempted_brokers)}")
        if self.failure_reasons:
            details.append(f"failures={'; '.join(self.failure_reasons)}")
        suffix = f" ({', '.join(details)})" if details else ""
        if self.broker_name:
            return f"[{self.broker_name}] {self.message}{suffix}"
        return f"{self.message}{suffix}"


__all__ = [
    "BrokerError",
    "BrokerConnectionError",
    "BrokerConfigurationError",
    "BrokerDataError",
    "BrokerOrderError",
    "BrokerTimeoutError",
    "NoBrokerAvailableError",
]
