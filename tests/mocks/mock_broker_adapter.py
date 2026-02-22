"""Mock broker adapter for deterministic offline integration tests."""

from __future__ import annotations

from datetime import UTC, datetime

from algotrading.src.broker import (
    AccountCallback,
    AccountInfo,
    BarData,
    BrokerAdapter,
    BrokerOrderError,
    ContractSpec,
    DataCallback,
    OrderCallback,
    OrderSide,
    OrderSpec,
    OrderStatus,
    PositionCallback,
    PositionInfo,
)


class MockBrokerAdapter(BrokerAdapter):
    """In-memory broker adapter with configurable responses.

    This mock avoids network calls and allows tests to control historical bars,
    account snapshots, position snapshots, and emitted real-time callbacks.
    """

    def __init__(self) -> None:
        self._connected = False
        self._account = AccountInfo(
            account_id="DU-MOCK",
            cash_balance=100_000.0,
            buying_power=200_000.0,
            currency="USD",
        )
        self._positions: list[PositionInfo] = []
        self._historical_bars: list[BarData] = []
        self._order_statuses: dict[str, OrderStatus] = {}
        self._order_id_seed = 1
        self._subscription_seed = 1000
        self._realtime_callbacks: dict[int, DataCallback] = {}
        self._order_callbacks: list[OrderCallback] = []
        self._account_callbacks: list[AccountCallback] = []
        self._position_callbacks: list[PositionCallback] = []
        self.connect_calls: list[tuple[str, int, int]] = []
        self.historical_calls: list[tuple[ContractSpec, datetime, str, str, str]] = []

    def set_historical_data(self, bars: list[BarData]) -> None:
        """Configure bars returned by ``request_historical_data``."""

        self._historical_bars = list(bars)

    def set_account_info(self, account: AccountInfo) -> None:
        """Configure account snapshot returned by ``get_account_info``."""

        self._account = account

    def set_positions(self, positions: list[PositionInfo]) -> None:
        """Configure position snapshot returned by ``get_positions``."""

        self._positions = list(positions)

    def emit_realtime_bar(
        self, bar: BarData, subscription_id: int | None = None
    ) -> None:
        """Emit a real-time bar through one or all subscriptions."""

        if subscription_id is not None:
            callback = self._realtime_callbacks.get(subscription_id)
            if callback is not None:
                callback(bar)
            return

        for callback in list(self._realtime_callbacks.values()):
            callback(bar)

    def connect(self, host: str, port: int, client_id: int) -> None:
        self.connect_calls.append((host, port, client_id))
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    @property
    def account_id(self) -> str:
        return self._account.account_id

    def request_historical_data(
        self,
        contract: ContractSpec,
        end_datetime: datetime,
        duration: str,
        bar_size: str,
        data_type: str = "TRADES",
    ) -> list[BarData]:
        self.historical_calls.append(
            (contract, end_datetime, duration, bar_size, data_type)
        )
        return list(self._historical_bars)

    def subscribe_realtime_data(
        self,
        contract: ContractSpec,
        bar_size: int,
        data_type: str,
        callback: DataCallback,
    ) -> int:
        del contract, bar_size, data_type
        sub_id = self._subscription_seed
        self._subscription_seed += 1
        self._realtime_callbacks[sub_id] = callback
        return sub_id

    def unsubscribe_realtime_data(self, subscription_id: int) -> None:
        self._realtime_callbacks.pop(subscription_id, None)

    def place_order(self, contract: ContractSpec, order: OrderSpec) -> str:
        del contract
        order_id = str(self._order_id_seed)
        self._order_id_seed += 1
        filled_quantity = (
            order.quantity if order.side in {OrderSide.BUY, OrderSide.SELL} else 0.0
        )
        status = OrderStatus(
            order_id=order_id,
            status="SUBMITTED",
            filled_quantity=0.0,
            remaining_quantity=filled_quantity,
            average_fill_price=None,
            last_update=datetime.now(tz=UTC),
        )
        self._order_statuses[order_id] = status
        for callback in self._order_callbacks:
            callback(status)
        return order_id

    def cancel_order(self, order_id: str) -> None:
        if order_id not in self._order_statuses:
            raise BrokerOrderError(
                message="Order not found",
                broker_name="mock",
                order_id=order_id,
            )

        previous = self._order_statuses[order_id]
        cancelled = OrderStatus(
            order_id=order_id,
            status="CANCELLED",
            filled_quantity=previous.filled_quantity,
            remaining_quantity=previous.remaining_quantity,
            average_fill_price=previous.average_fill_price,
            last_update=datetime.now(tz=UTC),
        )
        self._order_statuses[order_id] = cancelled
        for callback in self._order_callbacks:
            callback(cancelled)

    def get_order_status(self, order_id: str) -> OrderStatus:
        if order_id not in self._order_statuses:
            raise BrokerOrderError(
                message="Order not found",
                broker_name="mock",
                order_id=order_id,
            )
        return self._order_statuses[order_id]

    def register_order_callback(self, callback: OrderCallback) -> None:
        self._order_callbacks.append(callback)

    def get_account_info(self) -> AccountInfo:
        return self._account

    def get_positions(self) -> list[PositionInfo]:
        return list(self._positions)

    def subscribe_account_updates(self, callback: AccountCallback) -> None:
        self._account_callbacks.append(callback)
        callback(self._account)

    def subscribe_position_updates(self, callback: PositionCallback) -> None:
        self._position_callbacks.append(callback)
        for position in self._positions:
            callback(position)

    def get_next_order_id(self) -> str:
        return str(self._order_id_seed)
