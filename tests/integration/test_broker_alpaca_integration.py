"""Human-gated integration tests for Alpaca paper-trading adapter flows."""

from __future__ import annotations

import threading
from datetime import UTC, datetime

import pytest
from algotrading.src.broker import (
    AlpacaAdapter,
    BarData,
    ContractSpec,
    InstrumentType,
    OrderSide,
    OrderSpec,
    OrderType,
)


def _aapl_contract() -> ContractSpec:
    return ContractSpec(
        symbol="AAPL",
        instrument_type=InstrumentType.STOCK,
        exchange="SMART",
        currency="USD",
        primary_exchange="NASDAQ",
    )


@pytest.mark.requires_alpaca
def test_alpaca_adapter_connect_account_and_positions(
    confirm_alpaca_paper: dict[str, str],
) -> None:
    """Adapter connects to Alpaca paper trading and retrieves account snapshots."""

    pytest.importorskip("alpaca")
    adapter = AlpacaAdapter(config={**confirm_alpaca_paper, "paper": True})
    try:
        adapter.connect()
        account = adapter.get_account_info()
        positions = adapter.get_positions()

        assert adapter.is_connected()
        assert account.account_id
        assert account.currency
        assert positions is not None
    finally:
        adapter.disconnect()


@pytest.mark.requires_alpaca
def test_alpaca_adapter_historical_data(
    confirm_alpaca_paper: dict[str, str],
) -> None:
    """Adapter retrieves Alpaca historical bars in canonical ``BarData`` format."""

    pytest.importorskip("alpaca")
    adapter = AlpacaAdapter(config={**confirm_alpaca_paper, "paper": True})
    try:
        adapter.connect()
        bars = adapter.request_historical_data(
            contract=_aapl_contract(),
            end_datetime=datetime.now(tz=UTC),
            duration="1 D",
            bar_size="1 min",
        )

        assert bars
        assert isinstance(bars[-1], BarData)
        assert bars[-1].close > 0
    finally:
        adapter.disconnect()


@pytest.mark.requires_alpaca
def test_alpaca_adapter_place_and_cancel_paper_order(
    confirm_alpaca_paper: dict[str, str],
) -> None:
    """Adapter places and cancels a low-priced paper limit order."""

    pytest.importorskip("alpaca")
    adapter = AlpacaAdapter(config={**confirm_alpaca_paper, "paper": True})
    order_id: str | None = None
    try:
        adapter.connect()
        order_id = adapter.place_order(
            contract=_aapl_contract(),
            order=OrderSpec(
                side=OrderSide.BUY,
                quantity=1,
                order_type=OrderType.LIMIT,
                limit_price=0.01,
                time_in_force="DAY",
            ),
        )
        assert order_id
        adapter.cancel_order(order_id)
    finally:
        if order_id:
            try:
                adapter.cancel_order(order_id)
            except Exception:
                pass
        adapter.disconnect()


@pytest.mark.requires_alpaca
@pytest.mark.slow
def test_alpaca_adapter_realtime_subscription_with_historical_fallback(
    confirm_alpaca_paper: dict[str, str],
) -> None:
    """Adapter subscribes to real-time bars or validates data via fallback pull."""

    pytest.importorskip("alpaca")
    adapter = AlpacaAdapter(config={**confirm_alpaca_paper, "paper": True})
    event = threading.Event()
    received_closes: list[float] = []

    def on_bar(bar: BarData) -> None:
        received_closes.append(bar.close)
        event.set()

    subscription_id = -1
    try:
        adapter.connect()
        subscription_id = adapter.subscribe_realtime_data(
            contract=_aapl_contract(),
            bar_size=60,
            data_type="TRADES",
            callback=on_bar,
        )
        if event.wait(timeout=30):
            assert received_closes[0] > 0
            return

        bars = adapter.request_historical_data(
            contract=_aapl_contract(),
            end_datetime=datetime.now(tz=UTC),
            duration="1 D",
            bar_size="1 min",
        )
        assert bars
        assert bars[-1].close > 0
    finally:
        if subscription_id != -1:
            adapter.unsubscribe_realtime_data(subscription_id)
        adapter.disconnect()
