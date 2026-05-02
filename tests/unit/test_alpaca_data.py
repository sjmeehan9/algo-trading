"""Unit tests for Alpaca historical data helpers."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from algotrading.src.broker.adapters.alpaca.auth import AlpacaAuth
from algotrading.src.broker.adapters.alpaca.data import (
    AlpacaDataClient,
    ParsedTimeFrame,
    parse_bar_size,
    parse_duration_to_start,
)
from algotrading.src.broker.models import ContractSpec, InstrumentType


class _FakeDataClient:
    def __init__(self) -> None:
        self.requests: list[object] = []

    def get_stock_bars(self, request: object) -> dict[str, list[dict[str, object]]]:
        """Return deterministic fake bars for a request."""

        self.requests.append(request)
        return {
            "AAPL": [
                {
                    "timestamp": "2026-05-01T14:30:00Z",
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 1000,
                }
            ]
        }


def _stock_contract() -> ContractSpec:
    return ContractSpec(
        symbol="AAPL",
        instrument_type=InstrumentType.STOCK,
        exchange="SMART",
        currency="USD",
    )


def test_parse_duration_to_start_handles_ib_style_units() -> None:
    """Duration strings convert to UTC-aware start timestamps."""

    end = datetime(2026, 5, 1, 15, 0, tzinfo=UTC)

    assert parse_duration_to_start(end, "30 S") == datetime(
        2026, 5, 1, 14, 59, 30, tzinfo=UTC
    )
    assert parse_duration_to_start(end, "2 D") == datetime(
        2026, 4, 29, 15, 0, tzinfo=UTC
    )
    assert parse_duration_to_start(end.replace(tzinfo=None), "1 H").tzinfo is UTC


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1 min", ParsedTimeFrame(1, "Minute")),
        ("5 mins", ParsedTimeFrame(5, "Minute")),
        ("1 hour", ParsedTimeFrame(1, "Hour")),
        ("1 day", ParsedTimeFrame(1, "Day")),
        ("10 secs", ParsedTimeFrame(10, "Second")),
    ],
)
def test_parse_bar_size(value: str, expected: ParsedTimeFrame) -> None:
    """Bar-size strings convert into parsed Alpaca timeframe values."""

    assert parse_bar_size(value) == expected


def test_build_stock_bars_request_uses_configured_feed() -> None:
    """Historical requests include the configured Alpaca data feed."""

    pytest.importorskip("alpaca")
    wrapper = AlpacaDataClient(
        AlpacaAuth("key", "secret", data_feed="iex"),
    )
    request = wrapper._build_stock_bars_request(
        symbol="AAPL",
        start=datetime(2026, 5, 1, 14, 0, tzinfo=UTC),
        end=datetime(2026, 5, 1, 15, 0, tzinfo=UTC),
        timeframe=ParsedTimeFrame(1, "Minute"),
    )

    feed = getattr(request, "feed")
    assert getattr(feed, "value", feed) == "iex"


def test_data_client_returns_normalized_bars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Historical data wrapper returns normalized bars without SDK dependency."""

    data_client = _FakeDataClient()
    wrapper = AlpacaDataClient(
        AlpacaAuth("key", "secret"),
        data_client=data_client,
    )
    monkeypatch.setattr(
        wrapper,
        "_build_stock_bars_request",
        lambda symbol, start, end, timeframe: {
            "symbol": symbol,
            "start": start,
            "end": end,
            "timeframe": timeframe,
        },
    )

    bars = wrapper.get_historical_bars(
        contract=_stock_contract(),
        end_datetime=datetime(2026, 5, 1, 15, 0, tzinfo=UTC),
        duration="1 D",
        bar_size="1 min",
    )

    assert len(bars) == 1
    assert bars[0].close == 100.5
    assert data_client.requests[0]["symbol"] == "AAPL"
