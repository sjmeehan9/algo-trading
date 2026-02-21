"""Unit tests for Trading and OrderManager adapter refactor."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from algotrading.src.broker import (
    AccountInfo,
    BarData,
    BrokerAdapter,
    ContractSpec,
    InstrumentType,
    OrderSide,
    OrderSpec,
    OrderStatus,
    PositionInfo,
)
from algotrading.src.trading.order import OrderManager
from algotrading.src.trading.trading import Trading


class _PredictStub:
    def __init__(self, config: dict, pipeline: dict) -> None:
        del config, pipeline

    def get_action(self, state: dict) -> tuple[object, object]:
        del state

        class _Action:
            def item(self) -> int:
                return 0

        return _Action(), None


class _TimerStub:
    def __init__(self, seconds: int | float, callback, args=None):
        self.seconds = seconds
        self.callback = callback
        self.args = args or []
        self.started = False
        self.cancelled = False

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.cancelled = True


class _MockBrokerAdapter(BrokerAdapter):
    def __init__(self) -> None:
        self.connected = False
        self.connect_calls: list[tuple[str, int, int]] = []
        self.disconnect_calls = 0
        self.place_order_calls: list[tuple[ContractSpec, OrderSpec]] = []
        self.cancel_order_calls: list[str] = []
        self.order_callbacks = []
        self.account_callbacks = []
        self.position_callbacks = []
        self._last_order_id = "41"

        self._account_info = AccountInfo(
            account_id="DU123",
            cash_balance=20_000.0,
            buying_power=40_000.0,
            currency="USD",
        )
        self._positions: list[PositionInfo] = []

    def connect(self, host: str, port: int, client_id: int) -> None:
        self.connect_calls.append((host, port, client_id))
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False
        self.disconnect_calls += 1

    def is_connected(self) -> bool:
        return self.connected

    @property
    def account_id(self) -> str:
        return self._account_info.account_id

    def request_historical_data(
        self,
        contract: ContractSpec,
        end_datetime: datetime,
        duration: str,
        bar_size: str,
        data_type: str = "TRADES",
    ) -> list[BarData]:
        del contract, end_datetime, duration, bar_size, data_type
        return []

    def subscribe_realtime_data(
        self,
        contract: ContractSpec,
        bar_size: int,
        data_type: str,
        callback,
    ) -> int:
        del contract, bar_size, data_type, callback
        return 1

    def unsubscribe_realtime_data(self, subscription_id: int) -> None:
        del subscription_id

    def place_order(self, contract: ContractSpec, order: OrderSpec) -> str:
        self.place_order_calls.append((contract, order))
        return self._last_order_id

    def cancel_order(self, order_id: str) -> None:
        self.cancel_order_calls.append(order_id)

    def get_order_status(self, order_id: str) -> OrderStatus:
        return OrderStatus(
            order_id=order_id,
            status="SUBMITTED",
            filled_quantity=0.0,
            remaining_quantity=1.0,
            average_fill_price=None,
            last_update=datetime.now(tz=UTC),
        )

    def register_order_callback(self, callback) -> None:
        self.order_callbacks.append(callback)

    def get_account_info(self) -> AccountInfo:
        return self._account_info

    def get_positions(self) -> list[PositionInfo]:
        return list(self._positions)

    def subscribe_account_updates(self, callback) -> None:
        self.account_callbacks.append(callback)

    def subscribe_position_updates(self, callback) -> None:
        self.position_callbacks.append(callback)

    def get_next_order_id(self) -> str:
        return self._last_order_id


@pytest.fixture
def trading_instance(
    mock_config: dict,
    mock_pipeline_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Trading, _MockBrokerAdapter]:
    """Create a Trading instance with stubbed dependencies and mock adapter."""

    monkeypatch.setattr("algotrading.src.trading.trading.Predict", _PredictStub)
    monkeypatch.setattr("algotrading.src.trading.trading.Timer", _TimerStub)

    config = dict(mock_config)
    pipeline = dict(mock_pipeline_config)
    config["stream_data"] = "real"

    adapter = _MockBrokerAdapter()
    trading = Trading(config, pipeline, adapter=adapter)

    return trading, adapter


def test_order_manager_build_order_returns_order_spec(
    mock_config: dict,
    mock_pipeline_config: dict,
) -> None:
    """OrderManager returns broker-agnostic OrderSpec values."""

    manager = OrderManager(mock_config, mock_pipeline_config)

    order, active_pos = manager.buildOrder("BUY", 5)

    assert order.side == OrderSide.BUY
    assert order.quantity == 5
    assert active_pos == "BUY_PEND"


def test_trading_connects_and_registers_adapter_callbacks(
    trading_instance: tuple[Trading, _MockBrokerAdapter],
) -> None:
    """Trading startup should connect and subscribe via adapter APIs."""

    trading, adapter = trading_instance

    assert adapter.connect_calls
    assert len(adapter.order_callbacks) == 1
    assert len(adapter.account_callbacks) == 1
    assert len(adapter.position_callbacks) == 1
    assert trading.payload.cashbalance == 20_000.0


def test_trading_callbacks_update_payload_values(
    trading_instance: tuple[Trading, _MockBrokerAdapter],
) -> None:
    """Account and position callbacks should update payload state."""

    trading, adapter = trading_instance

    account_update = AccountInfo(
        account_id="DU123",
        cash_balance=11_500.0,
        buying_power=20_000.0,
        currency="USD",
    )
    position_update = PositionInfo(
        contract=ContractSpec(
            symbol="AMD",
            instrument_type=InstrumentType.STOCK,
            exchange="SMART",
            currency="USD",
            primary_exchange="NASDAQ",
        ),
        quantity=7.0,
        average_cost=100.0,
        market_value=707.0,
        unrealized_pnl=7.0,
    )

    adapter.account_callbacks[0](account_update)
    adapter.position_callbacks[0](position_update)

    assert trading.payload.cashbalance == 11_500.0
    assert trading.payload.openunits == 7.0


def test_execute_order_and_cancel_delegate_to_adapter(
    trading_instance: tuple[Trading, _MockBrokerAdapter],
) -> None:
    """Trading order methods should place and cancel through adapter."""

    trading, adapter = trading_instance

    trading.payload.cashbalance = 10_000.0
    trading.payload.openunits = 0.0
    trading.payload.action_str = "BUY"
    trading.payload.action_int = 1

    trading.order.calcOrderSpec = lambda *args, **kwargs: ["BUY", 3, "open"]

    trading.executeOrder()

    assert len(adapter.place_order_calls) == 1
    placed_contract, placed_order = adapter.place_order_calls[0]
    assert placed_contract.symbol == "AMD"
    assert placed_order.side == OrderSide.BUY

    status_submitted = OrderStatus(
        order_id=trading.oid,
        status="SUBMITTED",
        filled_quantity=0.0,
        remaining_quantity=3.0,
        average_fill_price=None,
        last_update=datetime.now(tz=UTC),
    )
    status_filled = OrderStatus(
        order_id=trading.oid,
        status="FILLED",
        filled_quantity=3.0,
        remaining_quantity=0.0,
        average_fill_price=101.25,
        last_update=datetime.now(tz=UTC),
    )

    adapter.order_callbacks[0](status_submitted)
    assert trading.payload.active_pos == "BUY_PEND"

    trading.stopCancel(trading.oid)
    assert adapter.cancel_order_calls[-1] == trading.oid

    adapter.order_callbacks[0](status_filled)
    assert trading.payload.active_pos == "BUY_FILL"

    trading.stop()
    assert adapter.disconnect_calls == 1
