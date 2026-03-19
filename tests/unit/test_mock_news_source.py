"""Unit tests for mock news data source and generator."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from threading import Event

import pytest
from algotrading.src.data_pipeline.sources.exceptions import DataSourceConnectionError
from algotrading.src.data_pipeline.sources.mock_news_source import (
    MockNewsSource,
    NewsGenerator,
)
from algotrading.src.data_pipeline.types import DataType, NewsQuery


def test_news_generator_returns_valid_news_record() -> None:
    """Generator produces a syntactically valid NewsRecord."""

    generator = NewsGenerator(seed=7)
    record = generator.generate(
        symbol="aapl",
        timestamp=datetime(2026, 3, 1, 14, 30, tzinfo=UTC),
    )

    assert record.source
    assert record.headline
    assert record.symbols == ["AAPL"]
    assert -1.0 <= (record.sentiment_score or 0.0) <= 1.0


def test_fetch_news_requires_connection() -> None:
    """Fetching news while disconnected raises a connection error."""

    source = MockNewsSource(symbols=["AAPL"], frequency_seconds=1)
    query = NewsQuery(
        start_time=datetime(2026, 3, 1, 14, 30, tzinfo=UTC),
        end_time=datetime(2026, 3, 1, 14, 35, tzinfo=UTC),
        symbols=["AAPL"],
    )

    with pytest.raises(DataSourceConnectionError):
        source.fetch_news(query)


def test_fetch_news_honors_symbol_window_and_limit() -> None:
    """Generated history respects symbol filters and query limit."""

    source = MockNewsSource(symbols=["AAPL", "MSFT"], frequency_seconds=60)
    source.connect()

    query = NewsQuery(
        start_time=datetime(2026, 3, 1, 14, 30, tzinfo=UTC),
        end_time=datetime(2026, 3, 1, 14, 40, tzinfo=UTC),
        symbols=["AAPL"],
        limit=3,
    )
    records = source.fetch_news(query)

    assert len(records) == 3
    assert all(record.symbols == ["AAPL"] for record in records)
    assert all(
        query.start_time <= record.timestamp <= query.end_time for record in records
    )


def test_fetch_batch_returns_news_text_batch() -> None:
    """Base NewsDataSource bridge returns DataBatch for NEWS_TEXT data."""

    source = MockNewsSource(symbols=["AAPL"], frequency_seconds=60)
    source.connect()

    start = datetime(2026, 3, 1, 14, 30, tzinfo=UTC)
    end = start + timedelta(minutes=5)
    batch = source.fetch_batch(symbol="AAPL", start=start, end=end)

    assert batch.data_type == DataType.NEWS_TEXT
    assert batch.symbol == "AAPL"
    assert len(batch) > 0


def test_subscribe_and_unsubscribe_streaming() -> None:
    """Subscription emits callback records and can be cleanly stopped."""

    source = MockNewsSource(symbols=["AAPL"], frequency_seconds=1)
    source.connect()

    received_event = Event()
    received: list[str] = []

    def on_news(news) -> None:
        received.append(news.news_id)
        received_event.set()

    subscription_id = source.subscribe_news(["AAPL"], on_news)
    assert received_event.wait(timeout=2.0)

    source.unsubscribe_news(subscription_id)
    assert len(received) >= 1


def test_available_symbols_and_categories() -> None:
    """Source exposes configured symbols and supported categories."""

    source = MockNewsSource(symbols=["AAPL", "MSFT"], frequency_seconds=1)

    assert source.get_available_symbols() == ["AAPL", "MSFT"]
    categories = source.get_available_categories()
    assert "earnings" in categories
    assert "market" in categories
