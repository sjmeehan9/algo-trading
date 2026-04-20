"""Unit tests for Benzinga news source adapter behavior."""

from __future__ import annotations

import sys
import types
from datetime import UTC, datetime, timedelta

import pytest
from algotrading.src.data_pipeline.sources.benzinga_news import BenzingaNewsSource
from algotrading.src.data_pipeline.sources.exceptions import (
    DataSourceConnectionError,
)
from algotrading.src.data_pipeline.types import NewsQuery


class _FakeBenzingaNewsClient:
    """Tiny SDK-like client used to test Benzinga adapter parsing."""

    instances: list["_FakeBenzingaNewsClient"] = []
    responses: list[object] = []

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.calls: list[dict[str, object]] = []
        self.__class__.instances.append(self)

    def news(self, **params: object) -> object:
        self.calls.append(dict(params))
        if self.__class__.responses:
            return self.__class__.responses.pop(0)
        return []


def _install_fake_benzinga_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install fake benzinga modules so import-time SDK dependency is mocked."""

    benzinga_module = types.ModuleType("benzinga")
    news_data_module = types.ModuleType("benzinga.news_data")
    news_data_module.News = _FakeBenzingaNewsClient
    benzinga_module.news_data = news_data_module

    monkeypatch.setitem(sys.modules, "benzinga", benzinga_module)
    monkeypatch.setitem(sys.modules, "benzinga.news_data", news_data_module)


def test_fetch_news_requires_connection() -> None:
    """fetch_news raises when provider source has not connected yet."""

    source = BenzingaNewsSource(api_key="test-key")
    query = NewsQuery(
        start_time=datetime.now(tz=UTC) - timedelta(hours=1),
        end_time=datetime.now(tz=UTC),
        symbols=["AAPL"],
    )

    with pytest.raises(DataSourceConnectionError):
        source.fetch_news(query)


def test_fetch_news_parses_articles_and_applies_keyword_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Benzinga adapter maps fields to NewsRecord and filters by keywords."""

    _FakeBenzingaNewsClient.instances = []
    _FakeBenzingaNewsClient.responses = [
        [{"title": "Health check", "created": "2026-04-01T14:00:00Z"}],
        [
            {
                "title": "Apple raises subscription guidance",
                "teaser": "Strong demand supports growth outlook",
                "created": "2026-04-01T15:00:00Z",
                "url": "https://example.com/aapl-1",
                "stocks": [{"name": "AAPL"}],
                "channels": [{"name": "guidance"}],
            },
            {
                "title": "Macro update for broad market",
                "teaser": "Rates remain unchanged",
                "created": "2026-04-01T15:01:00Z",
                "url": "https://example.com/macro-1",
                "stocks": [{"name": "SPY"}],
                "channels": [{"name": "market"}],
            },
        ],
    ]

    _install_fake_benzinga_sdk(monkeypatch)

    source = BenzingaNewsSource(api_key="test-key")
    source.connect()

    query = NewsQuery(
        start_time=datetime(2026, 4, 1, 14, 55, tzinfo=UTC),
        end_time=datetime(2026, 4, 1, 15, 10, tzinfo=UTC),
        symbols=["AAPL"],
        keywords=["guidance"],
        limit=10,
        include_body=False,
    )

    records = source.fetch_news(query)

    assert len(records) == 1
    record = records[0]
    assert record.headline == "Apple raises subscription guidance"
    assert record.body == "Strong demand supports growth outlook"
    assert record.symbols == ["AAPL"]
    assert record.categories == ["guidance"]
    assert record.sentiment_score is None
    assert record.news_id

    client = _FakeBenzingaNewsClient.instances[0]
    assert client.calls[1]["company_tickers"] == "AAPL"
    assert client.calls[1]["display_output"] == "abstract"
