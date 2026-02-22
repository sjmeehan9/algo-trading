"""Abstract broker adapter interface.

This module defines the contract that all broker implementations must satisfy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Protocol, runtime_checkable

from algotrading.src.broker.models import (
    AccountInfo,
    BarData,
    ContractSpec,
    OrderSpec,
    OrderStatus,
    PositionInfo,
)


@runtime_checkable
class DataCallback(Protocol):
    """Callback protocol for real-time bar updates."""

    def __call__(self, bar_data: BarData) -> None:
        """Handle a new bar update."""


@runtime_checkable
class OrderCallback(Protocol):
    """Callback protocol for order status updates."""

    def __call__(self, order_status: OrderStatus) -> None:
        """Handle an order status update."""


@runtime_checkable
class AccountCallback(Protocol):
    """Callback protocol for account updates."""

    def __call__(self, account_info: AccountInfo) -> None:
        """Handle an account update."""


@runtime_checkable
class PositionCallback(Protocol):
    """Callback protocol for position updates."""

    def __call__(self, position_info: PositionInfo) -> None:
        """Handle a position update."""


class BrokerAdapter(ABC):
    """Abstract contract for broker data and trading operations.

    Implementations may be internally synchronous or callback/event driven.
    Callback registration methods should be thread-safe and non-blocking.
    """

    @abstractmethod
    def connect(self, host: str, port: int, client_id: int) -> None:
        """Establish a broker connection.

        Args:
            host: Broker host endpoint.
            port: Broker network port.
            client_id: Broker client identifier.

        Raises:
            BrokerConnectionError: If connection fails.
        """

        raise NotImplementedError("connect must be implemented")

    @abstractmethod
    def disconnect(self) -> None:
        """Close the broker connection gracefully."""

        raise NotImplementedError("disconnect must be implemented")

    @abstractmethod
    def is_connected(self) -> bool:
        """Return whether the adapter is currently connected."""

        raise NotImplementedError("is_connected must be implemented")

    @property
    @abstractmethod
    def account_id(self) -> str:
        """Return the connected broker account identifier."""

        raise NotImplementedError("account_id must be implemented")

    @abstractmethod
    def request_historical_data(
        self,
        contract: ContractSpec,
        end_datetime: datetime,
        duration: str,
        bar_size: str,
        data_type: str = "TRADES",
    ) -> list[BarData]:
        """Request historical bars synchronously.

        Args:
            contract: Instrument to request.
            end_datetime: End of requested historical range.
            duration: Broker-native duration string.
            bar_size: Broker-native bar size string.
            data_type: Data source type, defaults to ``TRADES``.

        Returns:
            Ordered historical bars.

        Raises:
            BrokerDataError: If the request fails.
            BrokerTimeoutError: If the request times out.
        """

        raise NotImplementedError("request_historical_data must be implemented")

    @abstractmethod
    def subscribe_realtime_data(
        self,
        contract: ContractSpec,
        bar_size: int,
        data_type: str,
        callback: DataCallback,
    ) -> int:
        """Start real-time market data streaming.

        Args:
            contract: Instrument to subscribe.
            bar_size: Real-time bar size in seconds.
            data_type: Data source type (for example, ``TRADES``).
            callback: Callback receiving bar updates.

        Returns:
            Adapter-specific subscription identifier.
        """

        raise NotImplementedError("subscribe_realtime_data must be implemented")

    @abstractmethod
    def unsubscribe_realtime_data(self, subscription_id: int) -> None:
        """Stop a previously registered real-time data subscription."""

        raise NotImplementedError("unsubscribe_realtime_data must be implemented")

    @abstractmethod
    def place_order(self, contract: ContractSpec, order: OrderSpec) -> str:
        """Place an order.

        Args:
            contract: Instrument contract.
            order: Broker-agnostic order specification.

        Returns:
            Broker-assigned order identifier.

        Raises:
            BrokerOrderError: If order submission fails.
        """

        raise NotImplementedError("place_order must be implemented")

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        """Cancel a pending order by identifier."""

        raise NotImplementedError("cancel_order must be implemented")

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatus:
        """Return the latest known status for an order."""

        raise NotImplementedError("get_order_status must be implemented")

    @abstractmethod
    def register_order_callback(self, callback: OrderCallback) -> None:
        """Register a callback for asynchronous order updates."""

        raise NotImplementedError("register_order_callback must be implemented")

    @abstractmethod
    def get_account_info(self) -> AccountInfo:
        """Return the latest account information snapshot."""

        raise NotImplementedError("get_account_info must be implemented")

    @abstractmethod
    def get_positions(self) -> list[PositionInfo]:
        """Return all currently known positions."""

        raise NotImplementedError("get_positions must be implemented")

    @abstractmethod
    def subscribe_account_updates(self, callback: AccountCallback) -> None:
        """Register a callback for asynchronous account updates."""

        raise NotImplementedError("subscribe_account_updates must be implemented")

    @abstractmethod
    def subscribe_position_updates(self, callback: PositionCallback) -> None:
        """Register a callback for asynchronous position updates."""

        raise NotImplementedError("subscribe_position_updates must be implemented")

    @abstractmethod
    def get_next_order_id(self) -> str:
        """Return a unique order identifier for outbound order placement."""

        raise NotImplementedError("get_next_order_id must be implemented")


__all__ = [
    "AccountCallback",
    "BrokerAdapter",
    "DataCallback",
    "OrderCallback",
    "PositionCallback",
]
