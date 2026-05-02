"""Alpaca implementation of the broker adapter interface."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Mapping

from algotrading.src.broker.adapter import (
    AccountCallback,
    BrokerAdapter,
    DataCallback,
    OrderCallback,
    PositionCallback,
)
from algotrading.src.broker.adapters.alpaca.auth import AlpacaAuth
from algotrading.src.broker.adapters.alpaca.data import AlpacaDataClient
from algotrading.src.broker.adapters.alpaca.streaming import AlpacaStreamClient
from algotrading.src.broker.adapters.alpaca.trading import AlpacaTradingClient
from algotrading.src.broker.exceptions import (
    BrokerConnectionError,
    BrokerDataError,
    BrokerOrderError,
)
from algotrading.src.broker.models import (
    AccountInfo,
    BarData,
    ContractSpec,
    OrderSpec,
    OrderStatus,
    PositionInfo,
)

logger = logging.getLogger(__name__)


class AlpacaAdapter(BrokerAdapter):
    """Alpaca Trading API implementation of ``BrokerAdapter``."""

    BROKER_NAME = "alpaca"

    def __init__(
        self,
        config: Mapping[str, object] | None = None,
        auth: AlpacaAuth | None = None,
        data_client: object | None = None,
        trading_client: object | None = None,
        stream_client: object | None = None,
    ) -> None:
        """Initialize adapter configuration and lazy Alpaca clients."""

        self.config = dict(config or {})
        self.auth = auth or self._auth_from_config(self.config)
        self._data = AlpacaDataClient(self.auth, data_client=data_client)
        self._trading = AlpacaTradingClient(self.auth, trading_client=trading_client)
        self._stream = AlpacaStreamClient(self.auth, stream_client=stream_client)
        self._connected = False
        self._account_id = ""
        self._local_order_seed = 1
        self._order_statuses: dict[str, OrderStatus] = {}
        self._order_callbacks: list[OrderCallback] = []
        self._account_callbacks: list[AccountCallback] = []
        self._position_callbacks: list[PositionCallback] = []

    def connect(self, host: str = "", port: int = 0, client_id: int = 0) -> None:
        """Establish Alpaca connectivity by verifying the trading account."""

        del host, port, client_id
        try:
            account = self._trading.verify_connection()
        except Exception as exc:
            self._connected = False
            raise BrokerConnectionError(
                message="Failed to connect to Alpaca",
                broker_name=self.BROKER_NAME,
                host=self.auth.trading_base_url,
                port=443,
                reason=str(exc),
            ) from exc

        self._connected = True
        self._account_id = account.account_id
        self._emit_account(account)
        logger.info("Connected to Alpaca account %s", self._account_id)

    def disconnect(self) -> None:
        """Close Alpaca streaming resources and mark the adapter disconnected."""

        self._stream.stop()
        self._connected = False

    def is_connected(self) -> bool:
        """Return whether Alpaca account verification has succeeded."""

        return self._connected

    @property
    def account_id(self) -> str:
        """Return the connected Alpaca account identifier."""

        return self._account_id

    def request_historical_data(
        self,
        contract: ContractSpec,
        end_datetime: datetime,
        duration: str,
        bar_size: str,
        data_type: str = "TRADES",
    ) -> list[BarData]:
        """Request historical bars from Alpaca's Market Data API."""

        self._ensure_connected_for_operation("request_historical_data")
        try:
            return self._data.get_historical_bars(
                contract=contract,
                end_datetime=end_datetime,
                duration=duration,
                bar_size=bar_size,
                data_type=data_type,
            )
        except Exception as exc:
            raise BrokerDataError(
                message="Failed to request Alpaca historical data",
                broker_name=self.BROKER_NAME,
                contract=contract,
                reason=str(exc),
            ) from exc

    def subscribe_realtime_data(
        self,
        contract: ContractSpec,
        bar_size: int,
        data_type: str,
        callback: DataCallback,
    ) -> int:
        """Subscribe to Alpaca real-time stock bars."""

        self._ensure_connected_for_operation("subscribe_realtime_data")
        if data_type.upper() != "TRADES":
            raise BrokerDataError(
                message="Alpaca streaming bars support data_type='TRADES' only",
                broker_name=self.BROKER_NAME,
                contract=contract,
                reason=f"unsupported data_type={data_type}",
            )
        try:
            return self._stream.subscribe_bars(contract, bar_size, callback)
        except Exception as exc:
            raise BrokerDataError(
                message="Failed to subscribe to Alpaca real-time data",
                broker_name=self.BROKER_NAME,
                contract=contract,
                reason=str(exc),
            ) from exc

    def unsubscribe_realtime_data(self, subscription_id: int) -> None:
        """Stop a previously registered Alpaca stream subscription."""

        self._stream.unsubscribe(subscription_id)

    def place_order(self, contract: ContractSpec, order: OrderSpec) -> str:
        """Place an Alpaca paper or live order."""

        self._ensure_connected_for_operation("place_order")
        try:
            status = self._trading.submit_order(contract, order)
        except Exception as exc:
            raise BrokerOrderError(
                message="Failed to place Alpaca order",
                broker_name=self.BROKER_NAME,
                reason=str(exc),
                original_order=order,
            ) from exc

        self._order_statuses[status.order_id] = status
        self._emit_order(status)
        return status.order_id

    def cancel_order(self, order_id: str) -> None:
        """Cancel a pending Alpaca order."""

        self._ensure_connected_for_operation("cancel_order")
        try:
            self._trading.cancel_order(order_id)
        except Exception as exc:
            raise BrokerOrderError(
                message="Failed to cancel Alpaca order",
                broker_name=self.BROKER_NAME,
                order_id=order_id,
                reason=str(exc),
            ) from exc

        status = self._cancelled_status(order_id)
        self._order_statuses[order_id] = status
        self._emit_order(status)

    def get_order_status(self, order_id: str) -> OrderStatus:
        """Return the latest known Alpaca order status."""

        self._ensure_connected_for_operation("get_order_status")
        try:
            status = self._trading.get_order_status(order_id)
        except Exception:
            cached = self._order_statuses.get(order_id)
            if cached is None:
                raise BrokerOrderError(
                    message="Alpaca order status not found",
                    broker_name=self.BROKER_NAME,
                    order_id=order_id,
                    reason="no live or cached status available",
                )
            return cached
        self._order_statuses[order_id] = status
        return status

    def register_order_callback(self, callback: OrderCallback) -> None:
        """Register a callback for locally observed Alpaca order updates."""

        self._order_callbacks.append(callback)

    def get_account_info(self) -> AccountInfo:
        """Fetch current Alpaca account information."""

        self._ensure_connected_for_operation("get_account_info")
        try:
            account = self._trading.get_account_info()
        except Exception as exc:
            raise BrokerDataError(
                message="Failed to retrieve Alpaca account information",
                broker_name=self.BROKER_NAME,
                reason=str(exc),
            ) from exc
        self._account_id = account.account_id
        return account

    def get_positions(self) -> list[PositionInfo]:
        """Fetch current Alpaca positions."""

        self._ensure_connected_for_operation("get_positions")
        try:
            return self._trading.get_positions()
        except Exception as exc:
            raise BrokerDataError(
                message="Failed to retrieve Alpaca positions",
                broker_name=self.BROKER_NAME,
                reason=str(exc),
            ) from exc

    def subscribe_account_updates(self, callback: AccountCallback) -> None:
        """Register an account callback and emit the current snapshot if available."""

        self._account_callbacks.append(callback)
        if self.is_connected():
            callback(self.get_account_info())

    def subscribe_position_updates(self, callback: PositionCallback) -> None:
        """Register a position callback and emit current positions if available."""

        self._position_callbacks.append(callback)
        if self.is_connected():
            for position in self.get_positions():
                callback(position)

    def get_next_order_id(self) -> str:
        """Return a local pre-submit order identifier for interface compatibility."""

        order_id = f"alpaca-local-{self._local_order_seed}"
        self._local_order_seed += 1
        return order_id

    def _cancelled_status(self, order_id: str) -> OrderStatus:
        previous = self._order_statuses.get(order_id)
        return OrderStatus(
            order_id=order_id,
            status="CANCELLED",
            filled_quantity=previous.filled_quantity if previous else 0.0,
            remaining_quantity=0.0,
            average_fill_price=previous.average_fill_price if previous else None,
            last_update=datetime.now(tz=UTC),
        )

    def _emit_order(self, status: OrderStatus) -> None:
        for callback in list(self._order_callbacks):
            callback(status)

    def _emit_account(self, account: AccountInfo) -> None:
        for callback in list(self._account_callbacks):
            callback(account)

    def _ensure_connected_for_operation(self, operation_name: str) -> None:
        if self.is_connected():
            return
        raise BrokerConnectionError(
            message=f"Cannot run {operation_name} while disconnected",
            broker_name=self.BROKER_NAME,
            host=self.auth.trading_base_url,
            port=443,
            reason="adapter is not connected",
        )

    def _auth_from_config(self, config: Mapping[str, object]) -> AlpacaAuth:
        api_key = _first_config_value(config, "api_key", "alpaca_api_key")
        secret_key = _first_config_value(config, "secret_key", "alpaca_secret_key")
        paper_value = _first_config_value(config, "paper", "alpaca_paper")
        data_feed = _first_config_value(config, "data_feed", "alpaca_data_feed")

        if api_key is None or secret_key is None:
            env_auth = AlpacaAuth.from_env(paper=_parse_optional_bool(paper_value))
            if data_feed is not None:
                return AlpacaAuth(
                    api_key=env_auth.api_key,
                    secret_key=env_auth.secret_key,
                    paper=env_auth.paper,
                    data_feed=str(data_feed),
                )
            return env_auth

        return AlpacaAuth(
            api_key=str(api_key),
            secret_key=str(secret_key),
            paper=(
                _parse_optional_bool(paper_value) if paper_value is not None else True
            ),
            data_feed=str(data_feed or "iex"),
        )


def _first_config_value(config: Mapping[str, object], *keys: str) -> object | None:
    for key in keys:
        value = config.get(key)
        if value not in (None, ""):
            return value
    return None


def _parse_optional_bool(value: object | None) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


__all__ = ["AlpacaAdapter"]
