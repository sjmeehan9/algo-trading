"""Unit tests for broker-agnostic data models."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest
from algotrading.src.broker import (
    AccountInfo,
    BarData,
    ContractSpec,
    InstrumentType,
    OrderSide,
    OrderSpec,
    OrderStatus,
    OrderType,
    PositionInfo,
    bar_data_from_ib_bar,
    contract_spec_from_ib_contract,
    contract_spec_to_ib_fields,
    order_spec_to_ib_fields,
    order_status_from_ib_fields,
    parse_ib_timestamp,
)


def test_contract_spec_instantiates_for_each_instrument_type() -> None:
    """ContractSpec supports stock, option, future, crypto, and index instruments."""

    stock = ContractSpec(
        symbol="AAPL",
        instrument_type=InstrumentType.STOCK,
        exchange="SMART",
        currency="USD",
        primary_exchange="NASDAQ",
    )
    option = ContractSpec(
        symbol="AAPL",
        instrument_type=InstrumentType.OPTION,
        exchange="SMART",
        currency="USD",
        expiry="20261218",
        strike=200.0,
        right="C",
        multiplier=100,
    )
    future = ContractSpec(
        symbol="ES",
        instrument_type=InstrumentType.FUTURE,
        exchange="CME",
        currency="USD",
        expiry="202603",
        multiplier=50,
    )
    crypto = ContractSpec(
        symbol="BTC",
        instrument_type=InstrumentType.CRYPTO,
        exchange="PAXOS",
        currency="USD",
    )
    index_contract = ContractSpec(
        symbol="SPX",
        instrument_type=InstrumentType.INDEX,
        exchange="CBOE",
        currency="USD",
    )

    assert stock.instrument_type is InstrumentType.STOCK
    assert option.instrument_type is InstrumentType.OPTION
    assert future.instrument_type is InstrumentType.FUTURE
    assert crypto.instrument_type is InstrumentType.CRYPTO
    assert index_contract.instrument_type is InstrumentType.INDEX


@pytest.mark.parametrize(
    ("order_type", "limit_price", "stop_price"),
    [
        (OrderType.MARKET, None, None),
        (OrderType.LIMIT, 123.45, None),
        (OrderType.STOP, None, 120.0),
        (OrderType.STOP_LIMIT, 121.0, 120.0),
    ],
)
def test_order_spec_instantiates_for_each_order_type(
    order_type: OrderType,
    limit_price: float | None,
    stop_price: float | None,
) -> None:
    """OrderSpec supports market, limit, stop, and stop-limit types."""

    spec = OrderSpec(
        side=OrderSide.BUY,
        quantity=10,
        order_type=order_type,
        limit_price=limit_price,
        stop_price=stop_price,
        time_in_force="DAY",
    )

    assert spec.order_type is order_type
    assert spec.side is OrderSide.BUY


def test_frozen_models_are_immutable() -> None:
    """Frozen dataclasses reject attribute mutation."""

    contract = ContractSpec(
        symbol="MSFT",
        instrument_type=InstrumentType.STOCK,
        exchange="SMART",
        currency="USD",
    )
    bar = BarData(
        timestamp=datetime(2026, 2, 20, 14, 30, tzinfo=UTC),
        open=100.0,
        high=105.0,
        low=99.0,
        close=104.0,
        volume=1_000,
    )
    status = OrderStatus(
        order_id="1001",
        status="SUBMITTED",
        filled_quantity=0.0,
        remaining_quantity=10.0,
        average_fill_price=None,
        last_update=datetime(2026, 2, 20, 14, 31, tzinfo=UTC),
    )
    account = AccountInfo(
        account_id="DU12345",
        cash_balance=100_000.0,
        buying_power=400_000.0,
        currency="USD",
    )
    position = PositionInfo(
        contract=contract,
        quantity=10,
        average_cost=100.0,
        market_value=1_050.0,
        unrealized_pnl=50.0,
    )

    with pytest.raises(FrozenInstanceError):
        contract.symbol = "GOOG"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        bar.volume = 2_000  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        status.status = "FILLED"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        account.currency = "EUR"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        position.average_cost = 101.0  # type: ignore[misc]


def test_to_dict_serialization_methods() -> None:
    """Model serialization methods return JSON-compatible payloads."""

    contract = ContractSpec(
        symbol="TSLA",
        instrument_type=InstrumentType.STOCK,
        exchange="SMART",
        currency="USD",
        primary_exchange="NASDAQ",
    )
    bar = BarData(
        timestamp=datetime(2026, 2, 20, 14, 35, tzinfo=UTC),
        open=300.0,
        high=305.0,
        low=295.0,
        close=302.0,
        volume=12_345,
        vwap=301.5,
        trade_count=321,
    )
    order = OrderSpec(
        side=OrderSide.SELL,
        quantity=5,
        order_type=OrderType.LIMIT,
        limit_price=303.0,
        time_in_force="GTC",
    )
    status = OrderStatus(
        order_id="2002",
        status="PARTIAL",
        filled_quantity=2.0,
        remaining_quantity=3.0,
        average_fill_price=302.5,
        last_update=datetime(2026, 2, 20, 14, 36, tzinfo=UTC),
    )
    account = AccountInfo(
        account_id="DU88888",
        cash_balance=250_000.0,
        buying_power=1_000_000.0,
        currency="USD",
    )
    position = PositionInfo(
        contract=contract,
        quantity=5,
        average_cost=290.0,
        market_value=1_510.0,
        unrealized_pnl=60.0,
    )

    assert contract.to_dict()["instrument_type"] == "STOCK"
    assert bar.to_dict()["timestamp"] == "2026-02-20T14:35:00+00:00"
    assert order.to_dict()["order_type"] == "LIMIT"
    assert status.to_dict()["status"] == "PARTIAL"
    assert account.to_dict()["account_id"] == "DU88888"
    assert position.to_dict()["contract"]["symbol"] == "TSLA"


def test_contract_conversion_to_and_from_ib_fields() -> None:
    """Contract conversion helpers map cleanly between model and IB fields."""

    option = ContractSpec(
        symbol="AAPL",
        instrument_type=InstrumentType.OPTION,
        exchange="SMART",
        currency="USD",
        expiry="20261218",
        strike=210.0,
        right="P",
        multiplier=100.0,
    )

    ib_fields = contract_spec_to_ib_fields(option)

    assert ib_fields["secType"] == "OPT"
    assert ib_fields["lastTradeDateOrContractMonth"] == "20261218"
    assert ib_fields["strike"] == 210.0
    assert ib_fields["right"] == "P"

    recovered = contract_spec_from_ib_contract(ib_fields)
    assert recovered == option


def test_order_and_status_conversion_from_ib_fields() -> None:
    """Order and order-status helpers normalize IB-compatible payloads."""

    order_spec = OrderSpec(
        side=OrderSide.BUY,
        quantity=3,
        order_type=OrderType.STOP_LIMIT,
        limit_price=99.5,
        stop_price=100.0,
    )

    ib_order = order_spec_to_ib_fields(order_spec)
    assert ib_order["action"] == "BUY"
    assert ib_order["orderType"] == "STP LMT"
    assert ib_order["lmtPrice"] == 99.5
    assert ib_order["auxPrice"] == 100.0

    ib_status = {
        "orderId": "3003",
        "status": "PartiallyFilled",
        "filled": 1,
        "remaining": 2,
        "avgFillPrice": 99.75,
        "lastUpdate": "20260220 14:40:00",
    }
    model_status = order_status_from_ib_fields(ib_status)

    assert model_status.order_id == "3003"
    assert model_status.status == "PARTIAL"
    assert model_status.filled_quantity == 1.0
    assert model_status.remaining_quantity == 2.0
    assert model_status.average_fill_price == 99.75


def test_bar_conversion_from_ib_values() -> None:
    """Bar helper converts IB bar payloads into normalized BarData."""

    ib_bar = {
        "date": "20260220 14:45:00",
        "open": 150.0,
        "high": 151.0,
        "low": 149.5,
        "close": 150.25,
        "volume": 2500,
        "wap": 150.3,
        "barCount": 42,
    }

    bar = bar_data_from_ib_bar(ib_bar)

    assert bar.open == 150.0
    assert bar.high == 151.0
    assert bar.low == 149.5
    assert bar.close == 150.25
    assert bar.volume == 2500
    assert bar.vwap == 150.3
    assert bar.trade_count == 42
    assert bar.timestamp.tzinfo is not None


def test_parse_ib_timestamp_accepts_multiple_formats() -> None:
    """Timestamp parser handles IB string and epoch formats."""

    from_compact = parse_ib_timestamp("20260220 14:50:00")
    from_epoch = parse_ib_timestamp("1761000600")
    from_tz_named = parse_ib_timestamp("20260220 13:46:00 US/Eastern")

    assert from_compact.tzinfo is not None
    assert from_epoch.tzinfo is not None
    assert from_tz_named.tzinfo is not None
