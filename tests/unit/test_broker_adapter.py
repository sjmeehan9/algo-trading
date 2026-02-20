"""Unit tests for broker adapter interfaces and exceptions."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from algotrading.src.broker import (
    AccountInfo,
    BarData,
    BrokerAdapter,
    BrokerConnectionError,
    BrokerDataError,
    BrokerError,
    BrokerOrderError,
    BrokerTimeoutError,
    ContractSpec,
    InstrumentType,
    OrderSide,
    OrderSpec,
    OrderStatus,
    OrderType,
    PositionInfo,
)


class _BaseSuperHarness(BrokerAdapter):
    """Concrete harness that delegates each method to BrokerAdapter super."""

    def connect(self, host: str, port: int, client_id: int) -> None:
        super().connect(host, port, client_id)

    def disconnect(self) -> None:
        super().disconnect()

    def is_connected(self) -> bool:
        return super().is_connected()

    @property
    def account_id(self) -> str:
        return super().account_id

    def request_historical_data(
        self,
        contract: ContractSpec,
        end_datetime: datetime,
        duration: str,
        bar_size: str,
        data_type: str = "TRADES",
    ) -> list[BarData]:
        return super().request_historical_data(
            contract=contract,
            end_datetime=end_datetime,
            duration=duration,
            bar_size=bar_size,
            data_type=data_type,
        )

    def subscribe_realtime_data(
        self,
        contract: ContractSpec,
        bar_size: int,
        data_type: str,
        callback,
    ) -> int:
        return super().subscribe_realtime_data(
            contract=contract,
            bar_size=bar_size,
            data_type=data_type,
            callback=callback,
        )

    def unsubscribe_realtime_data(self, subscription_id: int) -> None:
        super().unsubscribe_realtime_data(subscription_id)

    def place_order(self, contract: ContractSpec, order: OrderSpec) -> str:
        return super().place_order(contract, order)

    def cancel_order(self, order_id: str) -> None:
        super().cancel_order(order_id)

    def get_order_status(self, order_id: str) -> OrderStatus:
        return super().get_order_status(order_id)

    def register_order_callback(self, callback) -> None:
        super().register_order_callback(callback)

    def get_account_info(self) -> AccountInfo:
        return super().get_account_info()

    def get_positions(self) -> list[PositionInfo]:
        return super().get_positions()

    def subscribe_account_updates(self, callback) -> None:
        super().subscribe_account_updates(callback)

    def subscribe_position_updates(self, callback) -> None:
        super().subscribe_position_updates(callback)

    def get_next_order_id(self) -> str:
        return super().get_next_order_id()


def _sample_contract() -> ContractSpec:
    return ContractSpec(
        symbol="AAPL",
        instrument_type=InstrumentType.STOCK,
        exchange="SMART",
        currency="USD",
    )


def _sample_order() -> OrderSpec:
    return OrderSpec(side=OrderSide.BUY, quantity=1, order_type=OrderType.MARKET)


def _sample_bar() -> BarData:
    return BarData(
        timestamp=datetime(2026, 2, 21, 10, 0, tzinfo=UTC),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=1_000,
    )


def _sample_status() -> OrderStatus:
    return OrderStatus(
        order_id="101",
        status="SUBMITTED",
        filled_quantity=0.0,
        remaining_quantity=1.0,
        average_fill_price=None,
        last_update=datetime(2026, 2, 21, 10, 1, tzinfo=UTC),
    )


def _sample_account() -> AccountInfo:
    return AccountInfo(
        account_id="DU12345",
        cash_balance=10_000.0,
        buying_power=40_000.0,
        currency="USD",
    )


def _sample_position() -> PositionInfo:
    return PositionInfo(
        contract=_sample_contract(),
        quantity=1.0,
        average_cost=100.0,
        market_value=100.5,
        unrealized_pnl=0.5,
    )


def test_broker_adapter_abc_cannot_be_instantiated_directly() -> None:
    """BrokerAdapter is abstract and cannot be instantiated."""

    with pytest.raises(TypeError):
        BrokerAdapter()


def test_abstract_methods_raise_not_implemented_when_delegating_to_super() -> None:
    """Abstract members raise NotImplementedError when not overridden with behavior."""

    adapter = _BaseSuperHarness()

    with pytest.raises(NotImplementedError):
        adapter.connect("127.0.0.1", 7497, 1)
    with pytest.raises(NotImplementedError):
        adapter.disconnect()
    with pytest.raises(NotImplementedError):
        adapter.is_connected()
    with pytest.raises(NotImplementedError):
        _ = adapter.account_id
    with pytest.raises(NotImplementedError):
        adapter.request_historical_data(
            contract=_sample_contract(),
            end_datetime=datetime(2026, 2, 21, 10, 0, tzinfo=UTC),
            duration="1 D",
            bar_size="1 min",
        )
    with pytest.raises(NotImplementedError):
        adapter.subscribe_realtime_data(
            contract=_sample_contract(),
            bar_size=5,
            data_type="TRADES",
            callback=lambda _bar: None,
        )
    with pytest.raises(NotImplementedError):
        adapter.unsubscribe_realtime_data(1)
    with pytest.raises(NotImplementedError):
        adapter.place_order(contract=_sample_contract(), order=_sample_order())
    with pytest.raises(NotImplementedError):
        adapter.cancel_order("101")
    with pytest.raises(NotImplementedError):
        adapter.get_order_status("101")
    with pytest.raises(NotImplementedError):
        adapter.register_order_callback(lambda _status: None)
    with pytest.raises(NotImplementedError):
        adapter.get_account_info()
    with pytest.raises(NotImplementedError):
        adapter.get_positions()
    with pytest.raises(NotImplementedError):
        adapter.subscribe_account_updates(lambda _account: None)
    with pytest.raises(NotImplementedError):
        adapter.subscribe_position_updates(lambda _position: None)
    with pytest.raises(NotImplementedError):
        adapter.get_next_order_id()


def test_callback_protocols_are_runtime_compatible() -> None:
    """Registered callbacks are structurally compatible at runtime."""

    from algotrading.src.broker.adapter import (
        AccountCallback,
        DataCallback,
        OrderCallback,
        PositionCallback,
    )

    def on_bar(bar: BarData) -> None:
        assert bar.close > 0

    def on_order(status: OrderStatus) -> None:
        assert status.status in {
            "PENDING",
            "SUBMITTED",
            "PARTIAL",
            "FILLED",
            "CANCELLED",
            "REJECTED",
        }

    def on_account(account: AccountInfo) -> None:
        assert account.account_id

    def on_position(position: PositionInfo) -> None:
        assert position.contract.symbol

    assert isinstance(on_bar, DataCallback)
    assert isinstance(on_order, OrderCallback)
    assert isinstance(on_account, AccountCallback)
    assert isinstance(on_position, PositionCallback)

    on_bar(_sample_bar())
    on_order(_sample_status())
    on_account(_sample_account())
    on_position(_sample_position())


def test_broker_error_base_attributes() -> None:
    """BrokerError stores message and broker name metadata."""

    error = BrokerError(message="Generic failure", broker_name="ib")
    assert error.message == "Generic failure"
    assert error.broker_name == "ib"
    assert str(error) == "[ib] Generic failure"


def test_connection_error_attributes() -> None:
    """BrokerConnectionError stores connection context details."""

    error = BrokerConnectionError(
        message="Connection failed",
        broker_name="ib",
        host="127.0.0.1",
        port=7497,
        reason="timeout",
    )
    assert error.host == "127.0.0.1"
    assert error.port == 7497
    assert error.reason == "timeout"
    assert "127.0.0.1:7497" in str(error)


def test_order_error_attributes() -> None:
    """BrokerOrderError stores order operation context details."""

    order = _sample_order()
    error = BrokerOrderError(
        message="Order rejected",
        broker_name="ib",
        order_id="3001",
        reason="insufficient buying power",
        original_order=order,
    )
    assert error.order_id == "3001"
    assert error.reason == "insufficient buying power"
    assert error.original_order == order
    assert "order_id=3001" in str(error)


def test_data_error_attributes() -> None:
    """BrokerDataError stores data request context details."""

    contract = _sample_contract()
    error = BrokerDataError(
        message="Historical data request failed",
        broker_name="ib",
        contract=contract,
        reason="no market data permission",
    )
    assert error.contract == contract
    assert error.reason == "no market data permission"
    assert "symbol=AAPL" in str(error)


def test_timeout_error_attributes() -> None:
    """BrokerTimeoutError stores operation timeout details."""

    error = BrokerTimeoutError(
        message="Operation timed out",
        broker_name="ib",
        operation="request_historical_data",
        timeout_seconds=10.0,
    )
    assert error.operation == "request_historical_data"
    assert error.timeout_seconds == 10.0
    assert "timeout=10.0s" in str(error)
