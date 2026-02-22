"""Interactive Brokers implementation of the broker adapter interface."""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime

from algotrading.src.broker.adapter import (
    AccountCallback,
    BrokerAdapter,
    DataCallback,
    OrderCallback,
    PositionCallback,
)
from algotrading.src.broker.exceptions import (
    BrokerConnectionError,
    BrokerDataError,
    BrokerOrderError,
    BrokerTimeoutError,
)
from algotrading.src.broker.ib_converters import (
    bar_from_ib,
    contract_to_ib,
    ib_to_contract,
    order_to_ib,
)
from algotrading.src.broker.models import (
    AccountInfo,
    BarData,
    ContractSpec,
    OrderSpec,
    OrderStatus,
    PositionInfo,
)
from ibapi.client import EClient
from ibapi.wrapper import EWrapper


class InteractiveBrokersAdapter(BrokerAdapter, EWrapper, EClient):
    """Interactive Brokers TWS/Gateway adapter.

    The adapter wraps IB's callback-based API and exposes the synchronous and
    callback registration methods defined by `BrokerAdapter`.
    """

    BROKER_NAME = "interactive_brokers"
    CONNECT_TIMEOUT_SECONDS = 10.0
    HISTORICAL_TIMEOUT_SECONDS = 30.0

    def __init__(self) -> None:
        """Initialize adapter state, callback registries, and caches."""

        EWrapper.__init__(self)
        EClient.__init__(self, wrapper=self)
        self.logger = logging.getLogger(__name__)

        self._lock = threading.RLock()
        self._network_thread: threading.Thread | None = None

        self._connected_host: str | None = None
        self._connected_port: int | None = None

        self._next_order_id: int | None = None
        self._next_request_id: int = 1_000
        self._connection_event = threading.Event()

        self._account_id: str = ""
        self._account_snapshot: AccountInfo | None = None
        self._account_currency: str | None = None
        self._cash_balance: float = 0.0
        self._buying_power: float = 0.0
        self._account_updates_requested = False

        self._positions: dict[str, PositionInfo] = {}
        self._order_status: dict[str, OrderStatus] = {}

        self._historical_buffers: dict[int, list[BarData]] = {}
        self._historical_events: dict[int, threading.Event] = {}
        self._historical_contracts: dict[int, ContractSpec] = {}
        self._request_errors: dict[int, str] = {}

        self._realtime_callbacks: dict[int, DataCallback] = {}
        self._order_callbacks: list[OrderCallback] = []
        self._account_callbacks: list[AccountCallback] = []
        self._position_callbacks: list[PositionCallback] = []

    def connect(self, host: str, port: int, client_id: int) -> None:
        """Establish connection to TWS/Gateway and wait for IB handshake."""

        self._connection_event.clear()
        with self._lock:
            self._connected_host = host
            self._connected_port = port

        try:
            EClient.connect(self, host, port, client_id)
        except Exception as exc:
            raise BrokerConnectionError(
                message="Failed to connect to Interactive Brokers",
                broker_name=self.BROKER_NAME,
                host=host,
                port=port,
                reason=str(exc),
            ) from exc

        self._start_network_loop()

        if not self._connection_event.wait(timeout=self.CONNECT_TIMEOUT_SECONDS):
            self.disconnect()
            raise BrokerConnectionError(
                message="Timed out waiting for IB nextValidId handshake",
                broker_name=self.BROKER_NAME,
                host=host,
                port=port,
                reason=f"timeout={self.CONNECT_TIMEOUT_SECONDS}s",
            )

    def disconnect(self) -> None:
        """Close the IB connection and release local request state."""

        with self._lock:
            historical_events = list(self._historical_events.values())
            self._historical_events.clear()
            self._historical_buffers.clear()
            self._historical_contracts.clear()
            self._request_errors.clear()
            self._realtime_callbacks.clear()

        for event in historical_events:
            event.set()

        if self.isConnected():
            if self._account_updates_requested:
                try:
                    self.reqAccountUpdates(False, self._account_id or "")
                except Exception:
                    self.logger.debug(
                        "Failed to disable reqAccountUpdates on disconnect"
                    )
            EClient.disconnect(self)
        self._account_updates_requested = False

    def is_connected(self) -> bool:
        """Return whether the adapter is connected to IB."""

        return bool(self.isConnected())

    @property
    def account_id(self) -> str:
        """Return the known account id from IB callbacks."""

        return self._account_id

    def request_historical_data(
        self,
        contract: ContractSpec,
        end_datetime: datetime,
        duration: str,
        bar_size: str,
        data_type: str = "TRADES",
    ) -> list[BarData]:
        """Request historical data synchronously through IB callbacks."""

        self._ensure_connected_for_operation("request_historical_data")

        request_id = self._next_req_id()
        completion_event = threading.Event()
        with self._lock:
            self._historical_events[request_id] = completion_event
            self._historical_buffers[request_id] = []
            self._historical_contracts[request_id] = contract
            self._request_errors.pop(request_id, None)

        contract_ib = contract_to_ib(contract)
        end_str = self._format_end_datetime(end_datetime)

        try:
            self.reqHistoricalData(
                request_id,
                contract_ib,
                end_str,
                duration,
                bar_size,
                data_type,
                1,
                1,
                False,
                [],
            )
        except Exception as exc:
            self._clear_historical_request(request_id)
            raise BrokerDataError(
                message="Failed to request historical data",
                broker_name=self.BROKER_NAME,
                contract=contract,
                reason=str(exc),
            ) from exc

        if not completion_event.wait(timeout=self.HISTORICAL_TIMEOUT_SECONDS):
            self._clear_historical_request(request_id)
            self.cancelHistoricalData(request_id)
            raise BrokerTimeoutError(
                message="Historical data request timed out",
                broker_name=self.BROKER_NAME,
                operation="request_historical_data",
                timeout_seconds=self.HISTORICAL_TIMEOUT_SECONDS,
            )

        with self._lock:
            error_message = self._request_errors.pop(request_id, None)
            bars = list(self._historical_buffers.pop(request_id, []))
            self._historical_events.pop(request_id, None)
            self._historical_contracts.pop(request_id, None)

        if error_message:
            raise BrokerDataError(
                message="Historical data request failed",
                broker_name=self.BROKER_NAME,
                contract=contract,
                reason=error_message,
            )

        return bars

    def subscribe_realtime_data(
        self,
        contract: ContractSpec,
        bar_size: int,
        data_type: str,
        callback: DataCallback,
    ) -> int:
        """Subscribe to real-time bars and dispatch via callback."""

        self._ensure_connected_for_operation("subscribe_realtime_data")
        request_id = self._next_req_id()

        with self._lock:
            self._realtime_callbacks[request_id] = callback

        try:
            self.reqRealTimeBars(
                request_id,
                contract_to_ib(contract),
                bar_size,
                data_type,
                True,
                [],
            )
        except Exception as exc:
            with self._lock:
                self._realtime_callbacks.pop(request_id, None)
            raise BrokerDataError(
                message="Failed to subscribe to real-time data",
                broker_name=self.BROKER_NAME,
                contract=contract,
                reason=str(exc),
            ) from exc

        return request_id

    def unsubscribe_realtime_data(self, subscription_id: int) -> None:
        """Cancel an active real-time bar subscription."""

        with self._lock:
            self._realtime_callbacks.pop(subscription_id, None)

        self.cancelRealTimeBars(subscription_id)

    def place_order(self, contract: ContractSpec, order: OrderSpec) -> str:
        """Place an IB order and return the assigned order id."""

        self._ensure_connected_for_operation("place_order")

        try:
            order_id = int(self.get_next_order_id())
            self.placeOrder(
                order_id, contract_to_ib(contract), order_to_ib(order, order_id)
            )
        except Exception as exc:
            raise BrokerOrderError(
                message="Failed to place order",
                broker_name=self.BROKER_NAME,
                order_id=None,
                reason=str(exc),
                original_order=order,
            ) from exc

        initial_status = OrderStatus(
            order_id=str(order_id),
            status="PENDING",
            filled_quantity=0.0,
            remaining_quantity=float(order.quantity),
            average_fill_price=None,
            last_update=datetime.now(tz=UTC),
        )
        with self._lock:
            self._order_status[str(order_id)] = initial_status

        return str(order_id)

    def cancel_order(self, order_id: str) -> None:
        """Cancel a previously submitted order."""

        try:
            self.cancelOrder(int(order_id), "")
        except TypeError:
            self.cancelOrder(int(order_id))
        except Exception as exc:
            raise BrokerOrderError(
                message="Failed to cancel order",
                broker_name=self.BROKER_NAME,
                order_id=order_id,
                reason=str(exc),
            ) from exc

    def get_order_status(self, order_id: str) -> OrderStatus:
        """Return the latest known order status."""

        with self._lock:
            status = self._order_status.get(order_id)

        if status is None:
            raise BrokerOrderError(
                message="Order status not found",
                broker_name=self.BROKER_NAME,
                order_id=order_id,
                reason="no cached status available",
            )
        return status

    def register_order_callback(self, callback: OrderCallback) -> None:
        """Register callback for order updates."""

        with self._lock:
            self._order_callbacks.append(callback)

    def get_account_info(self) -> AccountInfo:
        """Return the latest account snapshot from IB callbacks."""

        with self._lock:
            account = self._account_snapshot

        if account is None:
            raise BrokerDataError(
                message="Account snapshot unavailable",
                broker_name=self.BROKER_NAME,
                reason="no account updates received",
            )
        return account

    def get_positions(self) -> list[PositionInfo]:
        """Return the latest known positions."""

        with self._lock:
            return list(self._positions.values())

    def subscribe_account_updates(self, callback: AccountCallback) -> None:
        """Register callback for account snapshot updates."""

        with self._lock:
            self._account_callbacks.append(callback)
            snapshot = self._account_snapshot

        if snapshot is not None:
            callback(snapshot)
        self._ensure_account_updates_subscription()

    def subscribe_position_updates(self, callback: PositionCallback) -> None:
        """Register callback for position updates."""

        with self._lock:
            self._position_callbacks.append(callback)
            positions = list(self._positions.values())

        for position in positions:
            callback(position)
        self._ensure_account_updates_subscription()

    def get_next_order_id(self) -> str:
        """Return and advance the next available IB order id."""

        with self._lock:
            if self._next_order_id is None:
                raise BrokerOrderError(
                    message="Order id unavailable",
                    broker_name=self.BROKER_NAME,
                    reason="nextValidId not received",
                )
            next_id = self._next_order_id
            self._next_order_id += 1

        return str(next_id)

    def nextValidId(self, order_id: int) -> None:
        """IB callback: receive the next valid order id."""

        with self._lock:
            if self._next_order_id is None or order_id > self._next_order_id:
                self._next_order_id = order_id
        self._connection_event.set()

    def managedAccounts(self, accounts_list: str) -> None:
        """IB callback: receive managed account identifiers."""

        account_ids = [
            account.strip() for account in accounts_list.split(",") if account.strip()
        ]
        if not account_ids:
            return
        with self._lock:
            if not self._account_id:
                self._account_id = account_ids[0]
        self._ensure_account_updates_subscription(force=True)

    def historicalData(self, reqId: int, bar: object) -> None:
        """IB callback: append historical bar to request buffer."""

        with self._lock:
            buffer = self._historical_buffers.get(reqId)
        if buffer is None:
            return
        buffer.append(bar_from_ib(bar))

    def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:
        """IB callback: mark historical request complete."""

        del start, end
        with self._lock:
            event = self._historical_events.get(reqId)
        if event is not None:
            event.set()

    def realtimeBar(
        self,
        reqId: int,
        time: int,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        wap: float,
        count: int,
    ) -> None:
        """IB callback: dispatch real-time bar updates to subscribers."""

        with self._lock:
            callback = self._realtime_callbacks.get(reqId)
        if callback is None:
            return

        timestamp = datetime.fromtimestamp(time, tz=UTC)
        bar = bar_from_ib(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "wap": wap,
                "barCount": count,
            },
            timestamp=timestamp,
        )
        callback(bar)

    def orderStatus(
        self,
        orderId: int,
        status: str,
        filled: float,
        remaining: float,
        avgFillPrice: float,
        permId: int,
        parentId: int,
        lastFillPrice: float,
        clientId: int,
        whyHeld: str,
        mktCapPrice: float,
    ) -> None:
        """IB callback: update and broadcast order status."""

        del permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice

        status_upper = status.upper()
        status_map = {
            "PENDINGSUBMIT": "PENDING",
            "PRESUBMITTED": "SUBMITTED",
            "SUBMITTED": "SUBMITTED",
            "PARTIALLYFILLED": "PARTIAL",
            "FILLED": "FILLED",
            "CANCELLED": "CANCELLED",
            "INACTIVE": "REJECTED",
            "APICANCELLED": "CANCELLED",
            "API CANCELLED": "CANCELLED",
            "REJECTED": "REJECTED",
        }
        normalized_status = status_map.get(status_upper, "SUBMITTED")

        status_model = OrderStatus(
            order_id=str(orderId),
            status=normalized_status,
            filled_quantity=float(filled),
            remaining_quantity=float(remaining),
            average_fill_price=(float(avgFillPrice) if avgFillPrice else None),
            last_update=datetime.now(tz=UTC),
        )

        with self._lock:
            self._order_status[str(orderId)] = status_model
            callbacks = list(self._order_callbacks)

        for callback in callbacks:
            callback(status_model)

    def updateAccountValue(
        self,
        key: str,
        val: str,
        currency: str,
        accountName: str,
    ) -> None:
        """IB callback: update cached account snapshot and notify subscribers."""

        if key not in {"CashBalance", "BuyingPower"}:
            return

        try:
            numeric_value = float(val)
        except ValueError:
            self.logger.warning(
                "Skipping non-numeric account value | key=%s value=%s",
                key,
                val,
            )
            return

        with self._lock:
            if not self._account_id:
                self._account_id = accountName
            if self._account_currency is None:
                self._account_currency = currency
            if currency != self._account_currency:
                return

            if key == "CashBalance":
                self._cash_balance = numeric_value
            if key == "BuyingPower":
                self._buying_power = numeric_value

            account = AccountInfo(
                account_id=self._account_id or accountName,
                cash_balance=self._cash_balance,
                buying_power=self._buying_power,
                currency=currency,
            )
            self._account_snapshot = account
            callbacks = list(self._account_callbacks)

        for callback in callbacks:
            callback(account)

    def updatePortfolio(
        self,
        contract: object,
        position: float,
        marketPrice: float,
        marketValue: float,
        averageCost: float,
        unrealizedPNL: float,
        realizedPNL: float,
        accountName: str,
    ) -> None:
        """IB callback: update cached positions and notify subscribers."""

        del marketPrice, realizedPNL, accountName

        try:
            contract_spec = ib_to_contract(contract)
        except ValueError:
            self.logger.warning("Skipping unsupported IB contract in updatePortfolio")
            return

        position_info = PositionInfo(
            contract=contract_spec,
            quantity=float(position),
            average_cost=float(averageCost),
            market_value=float(marketValue),
            unrealized_pnl=float(unrealizedPNL),
        )

        position_key = self._position_key(contract_spec)
        with self._lock:
            self._positions[position_key] = position_info
            callbacks = list(self._position_callbacks)

        for callback in callbacks:
            callback(position_info)

    def error(
        self,
        reqId: int,
        errorCode: int,
        errorString: str,
        advancedOrderRejectJson: str = "",
    ) -> None:
        """IB callback: log errors and attach request-scoped failures."""

        del advancedOrderRejectJson
        self.logger.error(
            "IB API error | req_id=%s error_code=%s message=%s",
            reqId,
            errorCode,
            errorString,
        )

        informational_codes = {2104, 2106, 2107, 2108, 2119, 2158}
        if reqId >= 0 and errorCode not in informational_codes:
            with self._lock:
                self._request_errors[reqId] = f"{errorCode}: {errorString}"
                event = self._historical_events.get(reqId)
            if event is not None:
                event.set()

    def _start_network_loop(self) -> None:
        with self._lock:
            if self._network_thread and self._network_thread.is_alive():
                return
            self._network_thread = threading.Thread(target=self.run, daemon=True)
            self._network_thread.start()

    def _ensure_account_updates_subscription(self, force: bool = False) -> None:
        if not self.isConnected():
            return

        with self._lock:
            if self._account_updates_requested and not force:
                return
            account_code = self._account_id or ""

        try:
            self.reqAccountUpdates(True, account_code)
            with self._lock:
                self._account_updates_requested = True
        except Exception:
            self.logger.debug("Failed to start reqAccountUpdates subscription")

    def _next_req_id(self) -> int:
        with self._lock:
            request_id = self._next_request_id
            self._next_request_id += 1
        return request_id

    def _position_key(self, contract: ContractSpec) -> str:
        return "|".join(
            [
                contract.symbol,
                contract.instrument_type.value,
                contract.exchange,
                contract.currency,
                contract.expiry or "",
                str(contract.strike or ""),
                contract.right or "",
            ]
        )

    def _format_end_datetime(self, end_datetime: datetime) -> str:
        timestamp = end_datetime
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        else:
            timestamp = timestamp.astimezone(UTC)
        return f"{timestamp.strftime('%Y%m%d %H:%M:%S')} UTC"

    def _clear_historical_request(self, request_id: int) -> None:
        with self._lock:
            self._historical_events.pop(request_id, None)
            self._historical_buffers.pop(request_id, None)
            self._historical_contracts.pop(request_id, None)
            self._request_errors.pop(request_id, None)

    def _ensure_connected_for_operation(self, operation_name: str) -> None:
        if self.is_connected():
            return
        host = self._connected_host or "<unknown-host>"
        port = self._connected_port or 0
        raise BrokerConnectionError(
            message=f"Cannot run {operation_name} while disconnected",
            broker_name=self.BROKER_NAME,
            host=host,
            port=port,
            reason="adapter is not connected",
        )


__all__ = ["InteractiveBrokersAdapter"]
