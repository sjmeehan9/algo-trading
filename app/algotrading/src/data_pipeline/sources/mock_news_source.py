"""Mock news provider for development and integration testing."""

from __future__ import annotations

import random
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from itertools import count
from threading import Event, RLock, Thread
from uuid import uuid4

from algotrading.src.data_pipeline.sources.exceptions import (
    DataSourceConnectionError,
    DataValidationError,
)
from algotrading.src.data_pipeline.sources.news_source import NewsDataSource
from algotrading.src.data_pipeline.types import NewsQuery, NewsRecord


class NewsGenerator:
    """Utility that produces synthetic but realistic-looking news records."""

    HEADLINES: tuple[str, ...] = (
        "{symbol} reports quarterly earnings beat",
        "{symbol} announces strategic partnership",
        "Analysts upgrade {symbol} on growth outlook",
        "{symbol} faces regulatory scrutiny after filing",
        "{symbol} expands product roadmap in key segment",
        "Market update: {symbol} trades higher on volume spike",
        "{symbol} management issues updated guidance",
        "Institutional flows increase exposure to {symbol}",
    )

    CATEGORIES: tuple[str, ...] = (
        "earnings",
        "product",
        "analyst",
        "regulatory",
        "market",
        "guidance",
    )

    SOURCES: tuple[str, ...] = ("MockWire", "MockFinance", "MockMarkets")

    def __init__(self, seed: int | None = None) -> None:
        """Create a deterministic generator when ``seed`` is provided."""

        self._random = random.Random(seed)

    def generate(self, symbol: str, timestamp: datetime) -> NewsRecord:
        """Generate one synthetic news record.

        Args:
            symbol: Symbol associated with the article.
            timestamp: Publication timestamp.

        Returns:
            A generated ``NewsRecord``.
        """

        normalized_symbol = symbol.strip().upper()
        if not normalized_symbol:
            raise ValueError("symbol must be non-empty")

        headline = self._random.choice(self.HEADLINES).format(symbol=normalized_symbol)
        category = self._random.choice(self.CATEGORIES)
        source = self._random.choice(self.SOURCES)
        sentiment = round(self._random.uniform(-1.0, 1.0), 4)
        news_id = str(uuid4())

        body = (
            f"{headline}. Traders are monitoring {normalized_symbol} after fresh "
            f"{category} developments from {source}."
        )

        return NewsRecord(
            timestamp=timestamp,
            headline=headline,
            body=body,
            source=source,
            symbols=[normalized_symbol],
            categories=[category],
            sentiment_score=sentiment,
            url=f"https://mocknews.local/article/{news_id}",
            news_id=news_id,
        )


class MockNewsSource(NewsDataSource):
    """Stub ``NewsDataSource`` implementation for development and testing."""

    def __init__(
        self,
        symbols: list[str],
        frequency_seconds: int = 60,
        generator: NewsGenerator | None = None,
        source_id: str = "mock_news",
    ) -> None:
        """Initialize a mock provider.

        Args:
            symbols: Symbols this source can emit.
            frequency_seconds: Streaming emission interval in seconds.
            generator: Optional custom generator for testing.
            source_id: Source identifier reported to the pipeline.
        """

        cleaned_symbols = sorted(
            {value.strip().upper() for value in symbols if value.strip()}
        )
        if not cleaned_symbols:
            raise ValueError("symbols must include at least one non-empty symbol")
        if frequency_seconds <= 0:
            raise ValueError("frequency_seconds must be greater than 0")
        if not source_id.strip():
            raise ValueError("source_id must be non-empty")

        self._symbols = cleaned_symbols
        self._frequency_seconds = frequency_seconds
        self._generator = generator or NewsGenerator()
        self._source_id = source_id.strip()

        self._connected = False
        self._lock = RLock()
        self._subscriptions: dict[int, tuple[Event, Thread]] = {}
        self._subscription_ids = count(1)

    @property
    def source_id(self) -> str:
        """Return unique source identifier."""

        return self._source_id

    @property
    def is_connected(self) -> bool:
        """Return connection status for this source."""

        return self._connected

    def connect(self) -> None:
        """Mark the source as connected."""

        with self._lock:
            self._connected = True

    def disconnect(self) -> None:
        """Stop all subscriptions and mark the source disconnected."""

        with self._lock:
            subscription_ids = list(self._subscriptions.keys())

        for subscription_id in subscription_ids:
            self.unsubscribe_news(subscription_id)

        with self._lock:
            self._connected = False

    def fetch_news(self, query: NewsQuery) -> list[NewsRecord]:
        """Generate historical mock news for the requested query window."""

        if not self._connected:
            raise DataSourceConnectionError("MockNewsSource is not connected")

        target_symbols = self._resolve_target_symbols(query.symbols)
        keywords = [value.lower() for value in query.keywords]
        categories = {value.lower() for value in query.categories}

        generated: list[NewsRecord] = []
        current = query.start_time
        while current <= query.end_time and len(generated) < query.limit:
            for symbol in target_symbols:
                record = self._generator.generate(symbol=symbol, timestamp=current)
                if categories and not categories.intersection(
                    {value.lower() for value in record.categories}
                ):
                    continue
                if keywords:
                    searchable = f"{record.headline} {record.body or ''}".lower()
                    if not any(keyword in searchable for keyword in keywords):
                        continue
                generated.append(record)
                if len(generated) >= query.limit:
                    break
            current = current + timedelta(seconds=self._frequency_seconds)

        generated.sort(key=lambda record: record.timestamp)
        return generated

    def subscribe_news(
        self,
        symbols: list[str],
        callback: Callable[[NewsRecord], None],
    ) -> int:
        """Start background generation and callback delivery."""

        if not self._connected:
            raise DataSourceConnectionError("MockNewsSource is not connected")
        if not callable(callback):
            raise DataValidationError("callback must be callable")

        target_symbols = self._resolve_target_symbols(symbols)
        subscription_id = next(self._subscription_ids)
        stop_event = Event()
        worker = Thread(
            target=self._run_subscription,
            kwargs={
                "symbols": target_symbols,
                "callback": callback,
                "stop_event": stop_event,
            },
            name=f"mock-news-{subscription_id}",
            daemon=True,
        )

        with self._lock:
            self._subscriptions[subscription_id] = (stop_event, worker)

        worker.start()
        return subscription_id

    def unsubscribe_news(self, subscription_id: int) -> None:
        """Stop an active subscription."""

        with self._lock:
            subscription = self._subscriptions.pop(subscription_id, None)

        if subscription is None:
            return

        stop_event, worker = subscription
        stop_event.set()
        worker.join(timeout=max(self._frequency_seconds, 1))

    def get_available_categories(self) -> list[str]:
        """Return supported category values."""

        return list(NewsGenerator.CATEGORIES)

    def get_available_symbols(self) -> list[str]:
        """Return symbols configured for this source."""

        return list(self._symbols)

    def get_available_dates(self, symbol: str) -> list[date]:
        """Return available dates for symbol.

        Mock source generates data on demand and does not pre-index dates.
        """

        del symbol
        return []

    def _resolve_target_symbols(self, requested_symbols: list[str]) -> list[str]:
        """Resolve requested symbols against configured source symbols."""

        requested = [
            value.strip().upper() for value in requested_symbols if value.strip()
        ]
        if not requested:
            return list(self._symbols)

        unsupported = [value for value in requested if value not in self._symbols]
        if unsupported:
            raise DataValidationError(
                f"Unsupported symbols for mock source: {sorted(set(unsupported))}"
            )

        return requested

    def _run_subscription(
        self,
        symbols: list[str],
        callback: Callable[[NewsRecord], None],
        stop_event: Event,
    ) -> None:
        """Generate records periodically until stopped."""

        while not stop_event.is_set():
            now = datetime.now(tz=UTC)
            for symbol in symbols:
                callback(self._generator.generate(symbol=symbol, timestamp=now))
            stop_event.wait(self._frequency_seconds)
