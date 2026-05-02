"""Unit tests for Alpaca broker conversion helpers."""

from __future__ import annotations

from datetime import UTC, datetime

from algotrading.src.broker.adapters.alpaca.models import (
    alpaca_account_to_account_info,
    alpaca_bar_to_bar_data,
    alpaca_order_to_order_status,
    alpaca_position_to_position_info,
    contract_to_alpaca_symbol,
)
from algotrading.src.broker.models import ContractSpec, InstrumentType


def test_alpaca_bar_to_bar_data_from_mapping() -> None:
    """Alpaca bar mappings convert into validated ``BarData`` objects."""

    bar = alpaca_bar_to_bar_data(
        {
            "timestamp": "2026-05-01T14:30:00Z",
            "open": 100.0,
            "high": 101.0,
            "low": 99.5,
            "close": 100.5,
            "volume": 1200,
            "vwap": 100.25,
            "trade_count": 42,
        }
    )

    assert bar.timestamp == datetime(2026, 5, 1, 14, 30, tzinfo=UTC)
    assert bar.close == 100.5
    assert bar.volume == 1200
    assert bar.trade_count == 42


def test_contract_to_alpaca_symbol_supports_stock_option_and_crypto() -> None:
    """Contract symbols are normalized into Alpaca stock, OCC, and crypto forms."""

    stock = ContractSpec(
        symbol="aapl",
        instrument_type=InstrumentType.STOCK,
        exchange="SMART",
        currency="USD",
    )
    option = ContractSpec(
        symbol="AAPL",
        instrument_type=InstrumentType.OPTION,
        exchange="SMART",
        currency="USD",
        expiry="20261218",
        strike=200.0,
        right="C",
    )
    crypto = ContractSpec(
        symbol="BTC",
        instrument_type=InstrumentType.CRYPTO,
        exchange="PAXOS",
        currency="USD",
    )

    assert contract_to_alpaca_symbol(stock) == "AAPL"
    assert contract_to_alpaca_symbol(option) == "AAPL261218C00200000"
    assert contract_to_alpaca_symbol(crypto) == "BTC/USD"


def test_account_position_and_order_converters() -> None:
    """Account, position, and order payloads map into canonical broker models."""

    account = alpaca_account_to_account_info(
        {
            "account_number": "PA123456",
            "cash": "10000.50",
            "buying_power": "40000.25",
            "currency": "USD",
        }
    )
    position = alpaca_position_to_position_info(
        {
            "symbol": "AAPL",
            "asset_class": "us_equity",
            "qty": "3",
            "avg_entry_price": "190.25",
            "market_value": "600.00",
            "unrealized_pl": "29.25",
        }
    )
    order_status = alpaca_order_to_order_status(
        {
            "id": "order-1",
            "status": "partially_filled",
            "qty": "10",
            "filled_qty": "4",
            "filled_avg_price": "100.50",
            "updated_at": "2026-05-01T14:31:00Z",
        }
    )

    assert account.account_id == "PA123456"
    assert account.buying_power == 40000.25
    assert position.contract.symbol == "AAPL"
    assert position.quantity == 3.0
    assert order_status.status == "PARTIAL"
    assert order_status.remaining_quantity == 6.0
