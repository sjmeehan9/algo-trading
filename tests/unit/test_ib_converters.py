"""Unit tests for Interactive Brokers conversion helpers."""

from __future__ import annotations

from datetime import UTC, datetime

from algotrading.src.broker import (
    ContractSpec,
    InstrumentType,
    OrderSide,
    OrderSpec,
    OrderType,
    bar_from_ib,
    contract_to_ib,
    ib_to_contract,
    order_to_ib,
)
from ibapi.contract import Contract


def test_contract_to_ib_maps_stock_fields() -> None:
    """Stock contract maps to IB `Contract` fields correctly."""

    spec = ContractSpec(
        symbol="AAPL",
        instrument_type=InstrumentType.STOCK,
        exchange="SMART",
        currency="USD",
        primary_exchange="NASDAQ",
    )

    contract = contract_to_ib(spec)

    assert contract.symbol == "AAPL"
    assert contract.secType == "STK"
    assert contract.exchange == "SMART"
    assert contract.currency == "USD"
    assert contract.primaryExchange == "NASDAQ"


def test_ib_to_contract_maps_option_fields() -> None:
    """Option contract maps from IB `Contract` to `ContractSpec` correctly."""

    contract = Contract()
    contract.symbol = "AAPL"
    contract.secType = "OPT"
    contract.exchange = "SMART"
    contract.currency = "USD"
    contract.primaryExchange = "NASDAQ"
    contract.lastTradeDateOrContractMonth = "20261218"
    contract.strike = 220.0
    contract.right = "C"
    contract.multiplier = "100"

    spec = ib_to_contract(contract)

    assert spec.instrument_type is InstrumentType.OPTION
    assert spec.expiry == "20261218"
    assert spec.strike == 220.0
    assert spec.right == "C"
    assert spec.multiplier == 100.0


def test_order_to_ib_maps_and_sets_flags() -> None:
    """Order conversion applies type mappings and required IB flags."""

    order_spec = OrderSpec(
        side=OrderSide.BUY,
        quantity=5,
        order_type=OrderType.LIMIT,
        limit_price=101.5,
        time_in_force="GTC",
    )

    order = order_to_ib(order_spec, order_id=123)

    assert order.orderId == 123
    assert order.action == "BUY"
    assert order.totalQuantity == 5
    assert order.orderType == "LMT"
    assert order.lmtPrice == 101.5
    assert order.tif == "GTC"
    assert order.eTradeOnly is False
    assert order.firmQuoteOnly is False


def test_bar_from_ib_converts_mapping_payload() -> None:
    """Bar conversion normalizes IB fields to `BarData`."""

    bar = bar_from_ib(
        {
            "date": "20260220 14:45:00",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1500,
            "wap": 100.4,
            "barCount": 42,
        }
    )

    assert bar.open == 100.0
    assert bar.close == 100.5
    assert bar.volume == 1500
    assert bar.vwap == 100.4
    assert bar.trade_count == 42
    assert bar.timestamp.tzinfo is not None


def test_bar_from_ib_honors_explicit_timestamp() -> None:
    """Explicit timestamp override is used for realtime callback payloads."""

    timestamp = datetime(2026, 2, 21, 12, 0, tzinfo=UTC)
    bar = bar_from_ib(
        {
            "open": 50.0,
            "high": 52.0,
            "low": 49.5,
            "close": 51.0,
            "volume": 800,
            "wap": 50.8,
            "barCount": 11,
        },
        timestamp=timestamp,
    )

    assert bar.timestamp == timestamp
    assert bar.high == 52.0
