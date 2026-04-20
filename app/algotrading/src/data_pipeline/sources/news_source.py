"""Abstract interface for news/text data providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from datetime import date, datetime
from queue import Empty, Queue

from algotrading.src.data_pipeline.sources.base import DataSource
from algotrading.src.data_pipeline.sources.exceptions import DataValidationError
from algotrading.src.data_pipeline.types import (
    DataBatch,
    DataFrequency,
    DataRecord,
    DataType,
    NewsQuery,
    NewsRecord,
    SourceMetadata,
)


class NewsDataSource(DataSource, ABC):
    """Abstract base contract for news and text data providers.

    This interface extends ``DataSource`` with provider-agnostic methods used
    by downstream sentiment pipelines.
    """

    @property
    def metadata(self) -> SourceMetadata:
        """Return default metadata for news providers.

        Subclasses can override to augment runtime configuration details while
        preserving type/frequency capabilities.
        """

        return SourceMetadata(
            source_id=self.source_id,
            source_type="news",
            supported_types=[DataType.NEWS_TEXT],
            supported_frequencies=[DataFrequency.IRREGULAR],
            config={},
        )

    @abstractmethod
    def fetch_news(self, query: NewsQuery) -> list[NewsRecord]:
        """Query historical news records for the given filter criteria."""

    @abstractmethod
    def subscribe_news(
        self,
        symbols: list[str],
        callback: Callable[[NewsRecord], None],
    ) -> int:
        """Start provider subscription and return subscription identifier.

        Note:
            Many news providers do not support WebSocket/event streaming. Concrete
            implementations may use poll-based delivery with provider-specific
            intervals and latency characteristics.
        """

    @abstractmethod
    def unsubscribe_news(self, subscription_id: int) -> None:
        """Stop a previously created news subscription."""

    @abstractmethod
    def get_available_categories(self) -> list[str]:
        """Return categories supported by the provider."""

    def fetch_batch(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        data_type: DataType = DataType.NEWS_TEXT,
    ) -> DataBatch:
        """Fetch historical news and convert it into a generic ``DataBatch``."""

        if data_type != DataType.NEWS_TEXT:
            raise DataValidationError("NewsDataSource supports DataType.NEWS_TEXT only")
        if start > end:
            raise DataValidationError("start must be less than or equal to end")

        query = NewsQuery(start_time=start, end_time=end, symbols=[symbol])
        news_records = self.fetch_news(query)
        filtered = self.filter_by_relevance(news_records, symbol=symbol)
        pipeline_records = [self.news_to_record(news) for news in filtered]
        pipeline_records.sort(key=lambda record: record.timestamp)

        if pipeline_records:
            batch_start = pipeline_records[0].timestamp
            batch_end = pipeline_records[-1].timestamp
        else:
            batch_start = start
            batch_end = end

        return DataBatch(
            records=pipeline_records,
            start_time=batch_start,
            end_time=batch_end,
            data_type=DataType.NEWS_TEXT,
            symbol=symbol,
        )

    def fetch_stream(
        self,
        symbol: str,
        data_type: DataType = DataType.NEWS_TEXT,
    ) -> Iterator[DataRecord]:
        """Yield real-time news as generic ``DataRecord`` instances."""

        if data_type != DataType.NEWS_TEXT:
            raise DataValidationError("NewsDataSource supports DataType.NEWS_TEXT only")

        queue: Queue[NewsRecord] = Queue()

        def on_news(news: NewsRecord) -> None:
            queue.put(news)

        subscription_id = self.subscribe_news([symbol], on_news)
        try:
            while True:
                try:
                    news = queue.get(timeout=1.0)
                except Empty:
                    continue
                yield self.news_to_record(news)
        finally:
            self.unsubscribe_news(subscription_id)

    def news_to_record(self, news: NewsRecord) -> DataRecord:
        """Convert a provider ``NewsRecord`` to pipeline ``DataRecord``."""

        symbols = news.symbols or ["UNKNOWN"]
        return DataRecord(
            timestamp=news.timestamp,
            data_type=DataType.NEWS_TEXT,
            symbol=symbols[0],
            payload={
                "news_id": news.news_id,
                "headline": news.headline,
                "body": news.body,
                "source": news.source,
                "symbols": list(news.symbols),
                "categories": list(news.categories),
                "sentiment_score": news.sentiment_score,
                "url": news.url,
            },
            source_id=self.source_id,
            frequency=DataFrequency.IRREGULAR,
        )

    def filter_by_relevance(
        self,
        records: list[NewsRecord],
        symbol: str,
        min_relevance: float = 0.0,
    ) -> list[NewsRecord]:
        """Filter records by symbol relevance using lightweight heuristics.

        Relevance score rules:
        - 1.0 when symbol explicitly appears in ``record.symbols``.
        - 0.5 when symbol appears in headline/body text.
        - 0.0 otherwise.
        """

        normalized_symbol = symbol.strip().upper()
        if not normalized_symbol:
            return []
        if not 0.0 <= min_relevance <= 1.0:
            raise ValueError("min_relevance must be between 0.0 and 1.0")

        filtered: list[NewsRecord] = []
        for record in records:
            symbol_hits = [value.upper() for value in record.symbols]
            relevance = 0.0
            if normalized_symbol in symbol_hits:
                relevance = 1.0
            else:
                searchable_text = f"{record.headline} {record.body or ''}".upper()
                if normalized_symbol in searchable_text:
                    relevance = 0.5

            if relevance >= min_relevance:
                filtered.append(record)
        return filtered

    def get_available_dates(self, symbol: str) -> list[date]:
        """Return supported dates for ``symbol``.

        The abstract news interface does not provide date indexing; providers may
        override when date metadata is available.
        """

        del symbol
        return []

    def get_available_symbols(self) -> list[str]:
        """Return supported symbols for this provider.

        The abstract news interface does not require symbol discovery. Providers
        may override for richer capability reporting.
        """

        return []
