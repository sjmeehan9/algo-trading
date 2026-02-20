"""Integration tests for InteractiveBrokersAdapter against live IB paper account."""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime

import pytest
from algotrading.src.broker import (
    ContractSpec,
    InstrumentType,
    InteractiveBrokersAdapter,
    OrderSide,
    OrderSpec,
    OrderType,
)


def _amd_contract() -> ContractSpec:
    return ContractSpec(
        symbol="AMD",
        instrument_type=InstrumentType.STOCK,
        exchange="SMART",
        currency="USD",
        primary_exchange="NASDAQ",
    )


@pytest.mark.requires_ib
def test_ib_adapter_connect_and_historical_request(confirm_ib_gateway: dict) -> None:
    """Adapter connects to IB and returns historical bars."""

    adapter = InteractiveBrokersAdapter()
    conn = confirm_ib_gateway

    try:
        adapter.connect(conn["host"], conn["port"], conn["client_id"])
        assert adapter.is_connected()

        bars = adapter.request_historical_data(
            contract=_amd_contract(),
            end_datetime=datetime.now(tz=UTC),
            duration="1800 S",
            bar_size="10 secs",
            data_type="TRADES",
        )

        assert len(bars) > 0
        assert bars[-1].close > 0
    finally:
        adapter.disconnect()


@pytest.mark.requires_ib
def test_ib_adapter_realtime_subscription(confirm_ib_gateway: dict) -> None:
    """Adapter receives at least one real-time bar callback from IB."""

    adapter = InteractiveBrokersAdapter()
    conn = confirm_ib_gateway

    realtime_event = threading.Event()
    received_closes: list[float] = []

    def on_bar(bar) -> None:
        received_closes.append(bar.close)
        realtime_event.set()

    subscription_id = -1
    try:
        adapter.connect(conn["host"], conn["port"], conn["client_id"] + 1)
        subscription_id = adapter.subscribe_realtime_data(
            contract=_amd_contract(),
            bar_size=5,
            data_type="TRADES",
            callback=on_bar,
        )

        assert realtime_event.wait(timeout=45), (
            "Timed out waiting for real-time bar callback. Check market data "
            "entitlements and market session status."
        )
        assert received_closes[0] > 0
    finally:
        if subscription_id != -1:
            adapter.unsubscribe_realtime_data(subscription_id)
        adapter.disconnect()


@pytest.mark.requires_ib
def test_ib_adapter_place_and_cancel_order(confirm_ib_gateway: dict) -> None:
    """Adapter can place and cancel a paper order without raising."""

    adapter = InteractiveBrokersAdapter()
    conn = confirm_ib_gateway

    order_id: str | None = None
    try:
        adapter.connect(conn["host"], conn["port"], conn["client_id"] + 2)

        order_id = adapter.place_order(
            contract=_amd_contract(),
            order=OrderSpec(
                side=OrderSide.BUY,
                quantity=1,
                order_type=OrderType.LIMIT,
                limit_price=0.01,
                time_in_force="DAY",
            ),
        )

        time.sleep(1.5)
        adapter.cancel_order(order_id)
        status = adapter.get_order_status(order_id)
        assert status.order_id == order_id
    finally:
        if order_id is not None:
            try:
                adapter.cancel_order(order_id)
            except Exception:
                pass
        adapter.disconnect()
