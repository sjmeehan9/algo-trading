"""Unit tests for the Alpaca broker adapter."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from algotrading.src.broker.adapters.alpaca import (
    AlpacaAdapter,
    AlpacaAuth,
    AlpacaOrderPayload,
    AlpacaStreamClient,
    build_order_request,
)
from algotrading.src.broker.models import (
    BarData,
    ContractSpec,
    InstrumentType,
    OrderSide,
    OrderSpec,
    OrderType,
)


class _FakeTradingClient:
    def __init__(self) -> None:
        self.cancelled: list[str] = []
        self.orders: dict[str, dict[str, object]] = {}

    def get_account(self) -> dict[str, object]:
        """Return a deterministic account payload."""

        return {
            "account_number": "PA123",
            "cash": "10000",
            "buying_power": "40000",
            "currency": "USD",
        }

    def submit_order(self, request: object) -> dict[str, object]:
        """Record an order request and return a submitted order payload."""

        del request
        order = {
            "id": "order-1",
            "status": "new",
            "qty": "1",
            "filled_qty": "0",
            "updated_at": "2026-05-01T14:30:00Z",
        }
        self.orders["order-1"] = order
        return order

    def cancel_order_by_id(self, order_id: str) -> None:
        """Record a cancellation request."""

        self.cancelled.append(order_id)
        self.orders[order_id] = {
            "id": order_id,
            "status": "canceled",
            "qty": "1",
            "filled_qty": "0",
            "updated_at": "2026-05-01T14:31:00Z",
        }

    def get_order_by_id(self, order_id: str) -> dict[str, object]:
        """Return a cached order payload."""

        return self.orders[order_id]

    def get_all_positions(self) -> list[dict[str, object]]:
        """Return a deterministic position payload."""

        return [
            {
                "symbol": "AAPL",
                "asset_class": "us_equity",
                "qty": "2",
                "avg_entry_price": "100",
                "market_value": "210",
                "unrealized_pl": "10",
            }
        ]


class _FakeStream:
    def __init__(self) -> None:
        self.handlers: dict[str, object] = {}
        self.unsubscribed: list[str] = []
        self.stopped = False

    def subscribe_bars(self, handler: object, symbol: str) -> None:
        """Register a fake bar handler by symbol."""

        self.handlers[symbol] = handler

    def unsubscribe_bars(self, symbol: str) -> None:
        """Record symbol unsubscriptions."""

        self.unsubscribed.append(symbol)

    def run(self) -> None:
        """Return immediately to keep tests deterministic."""

    def stop(self) -> None:
        """Record stream shutdown."""

        self.stopped = True


def _stock_contract() -> ContractSpec:
    return ContractSpec(
        symbol="AAPL",
        instrument_type=InstrumentType.STOCK,
        exchange="SMART",
        currency="USD",
    )


def test_auth_headers_do_not_transform_credentials() -> None:
    """Auth headers use the canonical Alpaca header names."""

    headers = AlpacaAuth("key", "secret").get_headers()

    assert headers == {
        "APCA-API-KEY-ID": "key",
        "APCA-API-SECRET-KEY": "secret",
    }


def test_build_order_request_fallback_payload_for_limit_order() -> None:
    """Order request building exposes expected Alpaca fields offline."""

    request = build_order_request(
        _stock_contract(),
        OrderSpec(
            side=OrderSide.BUY,
            quantity=1,
            order_type=OrderType.LIMIT,
            limit_price=99.0,
        ),
    )

    assert getattr(request, "symbol") == "AAPL"
    assert getattr(request, "qty") == 1
    assert getattr(request, "limit_price") == 99.0
    if isinstance(request, AlpacaOrderPayload):
        assert request.order_type == "limit"


def test_alpaca_adapter_connect_order_account_and_positions() -> None:
    """Adapter delegates trading operations and normalizes responses."""

    adapter = AlpacaAdapter(
        config={"api_key": "key", "secret_key": "secret"},
        trading_client=_FakeTradingClient(),
    )
    order_statuses: list[str] = []
    adapter.register_order_callback(lambda status: order_statuses.append(status.status))

    adapter.connect()
    order_id = adapter.place_order(
        _stock_contract(),
        OrderSpec(side=OrderSide.BUY, quantity=1, order_type=OrderType.MARKET),
    )
    status = adapter.get_order_status(order_id)
    adapter.cancel_order(order_id)

    assert adapter.is_connected() is True
    assert adapter.account_id == "PA123"
    assert status.status == "SUBMITTED"
    assert adapter.get_account_info().cash_balance == 10000.0
    assert adapter.get_positions()[0].contract.symbol == "AAPL"
    assert order_statuses == ["SUBMITTED", "CANCELLED"]


def test_alpaca_adapter_historical_delegates_to_data_wrapper() -> None:
    """Adapter historical requests return wrapper-provided bars."""

    adapter = AlpacaAdapter(
        config={"api_key": "key", "secret_key": "secret"},
        trading_client=_FakeTradingClient(),
    )
    expected = [
        BarData(
            timestamp=datetime(2026, 5, 1, 14, 30, tzinfo=UTC),
            open=100,
            high=101,
            low=99,
            close=100.5,
            volume=1000,
        )
    ]
    adapter._data.get_historical_bars = lambda *args, **kwargs: expected

    adapter.connect()
    bars = adapter.request_historical_data(
        _stock_contract(),
        datetime(2026, 5, 1, 15, 0, tzinfo=UTC),
        "1 D",
        "1 min",
    )

    assert bars == expected


def test_alpaca_stream_client_dispatches_and_unsubscribes() -> None:
    """Stream wrapper dispatches Alpaca bar callbacks to subscribers."""

    fake_stream = _FakeStream()
    stream = AlpacaStreamClient(AlpacaAuth("key", "secret"), stream_client=fake_stream)
    received: list[float] = []

    subscription_id = stream.subscribe_bars(
        _stock_contract(),
        bar_size=60,
        callback=lambda bar: received.append(bar.close),
    )
    handler = fake_stream.handlers["AAPL"]
    asyncio.run(
        handler(
            {
                "symbol": "AAPL",
                "timestamp": "2026-05-01T14:30:00Z",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.25,
                "volume": 1000,
            }
        )
    )
    stream.unsubscribe(subscription_id)
    stream.stop()

    assert received == [100.25]
    assert fake_stream.unsubscribed == ["AAPL"]
    assert fake_stream.stopped is True
