"""Unit tests for NewsDataSource interface and news-specific pipeline types."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, datetime

import pytest
from algotrading.src.data_pipeline.sources.news_source import NewsDataSource
from algotrading.src.data_pipeline.types import (
    DataBatch,
    DataRecord,
    DataType,
    NewsQuery,
    NewsRecord,
)


class _ConcreteNewsSource(NewsDataSource):
    """Minimal concrete implementation for testing shared news-source logic."""

    def __init__(self) -> None:
        self._connected = False
        self._subscriptions: dict[int, Callable[[NewsRecord], None]] = {}
        self._next_subscription_id = 1

    @property
    def source_id(self) -> str:
        return "mock_news"

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def fetch_news(self, query: NewsQuery) -> list[NewsRecord]:
        del query
        return [
            NewsRecord(
                timestamp=datetime(2026, 2, 24, 14, 30, tzinfo=UTC),
                headline="AMD launches new accelerator",
                body="The AMD product line expands for AI workloads.",
                source="provider-a",
                symbols=["AMD", "NVDA"],
                categories=["technology", "earnings"],
                sentiment_score=0.7,
                url="https://example.com/news/1",
                news_id="n-1",
            )
        ]

    def subscribe_news(
        self,
        symbols: list[str],
        callback: Callable[[NewsRecord], None],
    ) -> int:
        del symbols
        subscription_id = self._next_subscription_id
        self._next_subscription_id += 1
        self._subscriptions[subscription_id] = callback

        callback(
            NewsRecord(
                timestamp=datetime(2026, 2, 24, 14, 31, tzinfo=UTC),
                headline="AMD guidance raised",
                body="Analysts revise expectations after new report.",
                source="provider-a",
                symbols=["AMD"],
                categories=["guidance"],
                sentiment_score=0.4,
                url="https://example.com/news/2",
                news_id="n-2",
            )
        )

        return subscription_id

    def unsubscribe_news(self, subscription_id: int) -> None:
        self._subscriptions.pop(subscription_id, None)

    def get_available_categories(self) -> list[str]:
        return ["technology", "earnings", "guidance"]


def test_news_data_source_is_abstract() -> None:
    """NewsDataSource cannot be instantiated directly."""

    with pytest.raises(TypeError):
        NewsDataSource()


def test_news_record_creation() -> None:
    """NewsRecord stores expected values for a valid payload."""

    record = NewsRecord(
        timestamp=datetime(2026, 2, 24, 14, 30, tzinfo=UTC),
        headline="Market update",
        body=None,
        source="provider-a",
        symbols=["AMD"],
        categories=["market"],
        sentiment_score=0.2,
        url=None,
        news_id="abc-1",
    )

    assert record.headline == "Market update"
    assert record.symbols == ["AMD"]


def test_news_query_validation() -> None:
    """NewsQuery validates time range and limit constraints."""

    with pytest.raises(ValueError, match="start_time"):
        NewsQuery(
            start_time=datetime(2026, 2, 24, 14, 31, tzinfo=UTC),
            end_time=datetime(2026, 2, 24, 14, 30, tzinfo=UTC),
        )

    with pytest.raises(ValueError, match="limit"):
        NewsQuery(
            start_time=datetime(2026, 2, 24, 14, 30, tzinfo=UTC),
            end_time=datetime(2026, 2, 24, 14, 31, tzinfo=UTC),
            limit=0,
        )


def test_news_to_record_conversion() -> None:
    """news_to_record converts NewsRecord into DataRecord payload."""

    source = _ConcreteNewsSource()
    news = NewsRecord(
        timestamp=datetime(2026, 2, 24, 14, 30, tzinfo=UTC),
        headline="AMD update",
        body="Body text",
        source="provider-a",
        symbols=["AMD", "NVDA"],
        categories=["market"],
        sentiment_score=0.5,
        url="https://example.com/news/3",
        news_id="n-3",
    )

    converted = source.news_to_record(news)

    assert isinstance(converted, DataRecord)
    assert converted.data_type == DataType.NEWS_TEXT
    assert converted.symbol == "AMD"
    assert converted.payload["news_id"] == "n-3"
    assert converted.payload["symbols"] == ["AMD", "NVDA"]


def test_fetch_batch_uses_news_query_and_relevance_filtering() -> None:
    """fetch_batch returns NEWS_TEXT batch for requested symbol window."""

    source = _ConcreteNewsSource()
    batch = source.fetch_batch(
        symbol="AMD",
        start=datetime(2026, 2, 24, 14, 0, tzinfo=UTC),
        end=datetime(2026, 2, 24, 15, 0, tzinfo=UTC),
    )

    assert isinstance(batch, DataBatch)
    assert batch.data_type == DataType.NEWS_TEXT
    assert batch.symbol == "AMD"
    assert len(batch) == 1


def test_fetch_stream_yields_news_records_and_unsubscribes() -> None:
    """fetch_stream yields converted records and unsubscribes on close."""

    source = _ConcreteNewsSource()
    stream: Iterator[DataRecord] = source.fetch_stream(symbol="AMD")

    first = next(stream)
    assert first.data_type == DataType.NEWS_TEXT
    assert first.symbol == "AMD"

    stream.close()
    assert source._subscriptions == {}


def test_filter_by_relevance_honors_threshold() -> None:
    """filter_by_relevance applies symbol and text relevance thresholds."""

    source = _ConcreteNewsSource()
    records = [
        NewsRecord(
            timestamp=datetime(2026, 2, 24, 14, 30, tzinfo=UTC),
            headline="AMD expands product roadmap",
            body="",
            source="provider-a",
            symbols=["AMD"],
            categories=["technology"],
            sentiment_score=0.1,
            url=None,
            news_id="n-4",
        ),
        NewsRecord(
            timestamp=datetime(2026, 2, 24, 14, 31, tzinfo=UTC),
            headline="Semiconductor rally continues",
            body="Analysts mention AMD by name",
            source="provider-a",
            symbols=["SOXX"],
            categories=["market"],
            sentiment_score=0.2,
            url=None,
            news_id="n-5",
        ),
    ]

    assert (
        len(source.filter_by_relevance(records, symbol="AMD", min_relevance=1.0)) == 1
    )
    assert (
        len(source.filter_by_relevance(records, symbol="AMD", min_relevance=0.5)) == 2
    )
