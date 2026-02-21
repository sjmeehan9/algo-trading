"""Integration tests for InteractiveBrokersAdapter against live IB paper account."""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime

import pytest
from algotrading.src.broker import (
    AccountInfo,
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
def test_ib_adapter_realtime_with_snapshot_fallback(confirm_ib_gateway: dict) -> None:
    """Prefer real-time callback, fallback to short historical snapshot-style pull.

    This path improves reliability outside market hours while still validating
    that market data retrieval works through the adapter.
    """

    adapter = InteractiveBrokersAdapter()
    conn = confirm_ib_gateway

    realtime_event = threading.Event()
    received_closes: list[float] = []

    def on_bar(bar) -> None:
        received_closes.append(bar.close)
        realtime_event.set()

    subscription_id = -1
    try:
        adapter.connect(conn["host"], conn["port"], conn["client_id"] + 4)

        subscription_id = adapter.subscribe_realtime_data(
            contract=_amd_contract(),
            bar_size=5,
            data_type="TRADES",
            callback=on_bar,
        )

        if realtime_event.wait(timeout=20):
            assert received_closes[0] > 0
            return

        bars = adapter.request_historical_data(
            contract=_amd_contract(),
            end_datetime=datetime.now(tz=UTC),
            duration="120 S",
            bar_size="1 min",
            data_type="TRADES",
        )
        assert bars, "Fallback historical snapshot-style request returned no bars."
        assert bars[-1].close > 0
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


@pytest.mark.requires_ib
def test_ib_adapter_account_balance_callback_delivery(
    confirm_ib_gateway: dict,
) -> None:
    """Adapter emits account callbacks with a valid cash-balance payload."""

    adapter = InteractiveBrokersAdapter()
    conn = confirm_ib_gateway

    account_event = threading.Event()
    received_accounts: list[AccountInfo] = []

    def on_account(account_info: AccountInfo) -> None:
        received_accounts.append(account_info)
        account_event.set()

    try:
        adapter.connect(conn["host"], conn["port"], conn["client_id"] + 3)
        adapter.subscribe_account_updates(on_account)

        assert account_event.wait(timeout=30), (
            "Timed out waiting for account callback. Check TWS/Gateway account "
            "update availability and API permissions."
        )

        latest_account = received_accounts[-1]
        assert latest_account.account_id
        assert latest_account.currency
        assert latest_account.cash_balance >= 0
    finally:
        adapter.disconnect()
