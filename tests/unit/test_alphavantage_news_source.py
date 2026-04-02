"""Unit tests for Alpha Vantage news source adapter behavior."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from algotrading.src.data_pipeline.sources.alphavantage_news import (
    AlphaVantageNewsSource,
)
from algotrading.src.data_pipeline.sources.exceptions import (
    DataSourceConnectionError,
    DataSourceError,
)
from algotrading.src.data_pipeline.types import NewsQuery


class _FakeResponse:
    """Lightweight response double for requests session mocking."""

    def __init__(self, payload: dict[str, object], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, object]:
        return self._payload


def test_connect_validates_api_key_with_provider_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """connect() sets source connected flag when provider check succeeds."""

    source = AlphaVantageNewsSource(api_key="test-key")

    def _fake_get(url: str, params: dict[str, object], timeout: int):
        del url, params, timeout
        return _FakeResponse(payload={"feed": []})

    monkeypatch.setattr(source._session, "get", _fake_get)
    source.connect()

    assert source.is_connected is True


def test_fetch_news_parses_feed_and_maps_sentiment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adapter parses Alpha Vantage feed payload into NewsRecord objects."""

    source = AlphaVantageNewsSource(api_key="test-key")

    responses = [
        _FakeResponse(payload={"feed": []}),
        _FakeResponse(
            payload={
                "feed": [
                    {
                        "time_published": "20260401T153000",
                        "title": "AAPL demand remains resilient",
                        "summary": "Management commentary remains positive",
                        "source": "Alpha Vantage",
                        "url": "https://example.com/aapl-1",
                        "overall_sentiment_score": "0.66",
                        "ticker_sentiment": [{"ticker": "AAPL"}],
                        "topics": [{"topic": "technology"}],
                    }
                ]
            }
        ),
    ]

    def _fake_get(url: str, params: dict[str, object], timeout: int):
        del url, params, timeout
        return responses.pop(0)

    monkeypatch.setattr(source._session, "get", _fake_get)
    source.connect()

    query = NewsQuery(
        start_time=datetime(2026, 4, 1, 15, 0, tzinfo=UTC),
        end_time=datetime(2026, 4, 1, 16, 0, tzinfo=UTC),
        symbols=["AAPL"],
        categories=["technology"],
        limit=5,
    )

    records = source.fetch_news(query)

    assert len(records) == 1
    record = records[0]
    assert record.headline == "AAPL demand remains resilient"
    assert record.sentiment_score == 0.66
    assert record.symbols == ["AAPL"]
    assert record.categories == ["technology"]
    assert record.news_id


def test_fetch_news_raises_on_provider_error_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider API error payloads are surfaced as DataSourceError."""

    source = AlphaVantageNewsSource(api_key="test-key")

    responses = [
        _FakeResponse(payload={"feed": []}),
        _FakeResponse(payload={"Error Message": "Invalid API call"}),
    ]

    def _fake_get(url: str, params: dict[str, object], timeout: int):
        del url, params, timeout
        return responses.pop(0)

    monkeypatch.setattr(source._session, "get", _fake_get)
    source.connect()

    query = NewsQuery(
        start_time=datetime.now(tz=UTC) - timedelta(hours=2),
        end_time=datetime.now(tz=UTC),
        symbols=["AAPL"],
    )

    with pytest.raises(DataSourceError):
        source.fetch_news(query)


def test_connect_raises_for_error_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """connect() fails fast when provider returns an API error payload."""

    source = AlphaVantageNewsSource(api_key="bad-key")

    def _fake_get(url: str, params: dict[str, object], timeout: int):
        del url, params, timeout
        return _FakeResponse(payload={"Error Message": "bad key"})

    monkeypatch.setattr(source._session, "get", _fake_get)

    with pytest.raises(DataSourceConnectionError):
        source.connect()
