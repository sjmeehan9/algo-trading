"""Unit tests for the Interactive Brokers broker adapter."""

from __future__ import annotations

from datetime import UTC, datetime

from algotrading.src.broker import (
    ContractSpec,
    InstrumentType,
    InteractiveBrokersAdapter,
    OrderSide,
    OrderSpec,
    OrderType,
)
from ibapi.contract import Contract


def _sample_contract() -> ContractSpec:
    return ContractSpec(
        symbol="AMD",
        instrument_type=InstrumentType.STOCK,
        exchange="SMART",
        currency="USD",
        primary_exchange="NASDAQ",
    )


def test_connect_completes_after_next_valid_id(monkeypatch) -> None:
    """Connect succeeds when `nextValidId` is received."""

    adapter = InteractiveBrokersAdapter()

    def fake_connect(self, host: str, port: int, client_id: int) -> None:
        del host, port, client_id
        adapter.nextValidId(900)

    monkeypatch.setattr("ibapi.client.EClient.connect", fake_connect)
    monkeypatch.setattr(adapter, "_start_network_loop", lambda: None)

    adapter.connect("127.0.0.1", 7497, 11)

    assert adapter.get_next_order_id() == "900"


def test_request_historical_data_collects_callback_bars(monkeypatch) -> None:
    """Historical requests aggregate callback bars and return normalized data."""

    adapter = InteractiveBrokersAdapter()
    monkeypatch.setattr(adapter, "is_connected", lambda: True)

    def fake_request(
        req_id: int,
        contract,
        end_datetime: str,
        duration: str,
        bar_size: str,
        data_type: str,
        use_rth: int,
        format_date: int,
        keep_up_to_date: bool,
        chart_options: list,
    ) -> None:
        del (
            contract,
            end_datetime,
            duration,
            bar_size,
            data_type,
            use_rth,
            format_date,
            keep_up_to_date,
            chart_options,
        )
        adapter.historicalData(
            req_id,
            {
                "date": "20260220 14:45:00",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000,
                "wap": 100.4,
                "barCount": 33,
            },
        )
        adapter.historicalDataEnd(req_id, "", "")

    monkeypatch.setattr(adapter, "reqHistoricalData", fake_request)

    bars = adapter.request_historical_data(
        contract=_sample_contract(),
        end_datetime=datetime(2026, 2, 21, 14, 30, tzinfo=UTC),
        duration="1 D",
        bar_size="5 secs",
    )

    assert len(bars) == 1
    assert bars[0].close == 100.5


def test_realtime_subscription_dispatches_callback(monkeypatch) -> None:
    """Realtime callback is invoked for bars on active subscription."""

    adapter = InteractiveBrokersAdapter()
    monkeypatch.setattr(adapter, "is_connected", lambda: True)

    requested_ids: list[int] = []
    cancelled_ids: list[int] = []
    received_closes: list[float] = []

    def fake_req_realtime(
        req_id: int,
        contract,
        bar_size: int,
        data_type: str,
        use_rth: bool,
        options: list,
    ) -> None:
        del contract, bar_size, data_type, use_rth, options
        requested_ids.append(req_id)

    def fake_cancel_realtime(req_id: int) -> None:
        cancelled_ids.append(req_id)

    monkeypatch.setattr(adapter, "reqRealTimeBars", fake_req_realtime)
    monkeypatch.setattr(adapter, "cancelRealTimeBars", fake_cancel_realtime)

    subscription_id = adapter.subscribe_realtime_data(
        contract=_sample_contract(),
        bar_size=5,
        data_type="TRADES",
        callback=lambda bar: received_closes.append(bar.close),
    )

    adapter.realtimeBar(subscription_id, 1761000600, 1.0, 2.0, 0.5, 1.5, 100, 1.4, 10)
    adapter.unsubscribe_realtime_data(subscription_id)

    assert requested_ids == [subscription_id]
    assert cancelled_ids == [subscription_id]
    assert received_closes == [1.5]


def test_place_order_and_order_status_flow(monkeypatch) -> None:
    """Placed orders are cached and updated by `orderStatus` callbacks."""

    adapter = InteractiveBrokersAdapter()
    adapter.nextValidId(300)
    monkeypatch.setattr(adapter, "is_connected", lambda: True)

    placed_order_ids: list[int] = []
    callback_statuses: list[str] = []

    def fake_place_order(order_id: int, contract, order) -> None:
        del contract, order
        placed_order_ids.append(order_id)

    monkeypatch.setattr(adapter, "placeOrder", fake_place_order)
    adapter.register_order_callback(
        lambda status: callback_statuses.append(status.status)
    )

    order_id = adapter.place_order(
        contract=_sample_contract(),
        order=OrderSpec(side=OrderSide.BUY, quantity=2, order_type=OrderType.MARKET),
    )

    pending = adapter.get_order_status(order_id)
    assert pending.status == "PENDING"
    assert placed_order_ids == [300]

    adapter.orderStatus(300, "Filled", 2.0, 0.0, 101.25, 1, 0, 101.25, 1, "", 0.0)
    updated = adapter.get_order_status(order_id)

    assert updated.status == "FILLED"
    assert callback_statuses == ["FILLED"]


def test_account_and_position_updates_cache_state() -> None:
    """Account and portfolio callbacks populate adapter state snapshots."""

    adapter = InteractiveBrokersAdapter()
    account_snapshots: list[float] = []
    position_symbols: list[str] = []

    adapter.subscribe_account_updates(
        lambda account: account_snapshots.append(account.cash_balance)
    )
    adapter.subscribe_position_updates(
        lambda position: position_symbols.append(position.contract.symbol)
    )

    adapter.updateAccountValue("CashBalance", "25000", "USD", "DU123")
    adapter.updateAccountValue("BuyingPower", "100000", "USD", "DU123")

    account = adapter.get_account_info()
    assert account.account_id == "DU123"
    assert account.cash_balance == 25000.0
    assert account.buying_power == 100000.0
    assert account_snapshots[-1] == 25000.0

    contract = Contract()
    contract.symbol = "AMD"
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"
    contract.primaryExchange = "NASDAQ"

    adapter.updatePortfolio(
        contract,
        3.0,
        102.0,
        306.0,
        100.0,
        6.0,
        0.0,
        "DU123",
    )

    positions = adapter.get_positions()
    assert len(positions) == 1
    assert positions[0].contract.symbol == "AMD"
    assert positions[0].market_value == 306.0
    assert position_symbols == ["AMD"]
