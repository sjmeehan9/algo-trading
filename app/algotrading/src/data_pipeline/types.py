"""Core type definitions for the standalone data pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Iterator

import pandas as pd


class DataType(str, Enum):
    """Supported data payload categories flowing through the pipeline."""

    MARKET_BAR = "MARKET_BAR"
    NEWS_TEXT = "NEWS_TEXT"
    INDICATOR = "INDICATOR"
    SIGNAL = "SIGNAL"


class DataFrequency(str, Enum):
    """Supported data update frequencies for records and batches."""

    TICK = "TICK"
    SECOND_1 = "SECOND_1"
    SECOND_5 = "SECOND_5"
    SECOND_10 = "SECOND_10"
    SECOND_30 = "SECOND_30"
    MINUTE_1 = "MINUTE_1"
    MINUTE_5 = "MINUTE_5"
    MINUTE_15 = "MINUTE_15"
    HOUR_1 = "HOUR_1"
    DAY_1 = "DAY_1"
    IRREGULAR = "IRREGULAR"


@dataclass(frozen=True, slots=True)
class DataRecord:
    """Single immutable record in the data pipeline.

    Args:
        timestamp: Event time for the record.
        data_type: Category describing payload semantics.
        symbol: Instrument or entity identifier.
        payload: Flexible payload for source-specific content.
        source_id: Optional source identifier.
        frequency: Optional frequency associated with this record.
    """

    timestamp: datetime
    data_type: DataType
    symbol: str
    payload: dict[str, object]
    source_id: str | None = None
    frequency: DataFrequency | None = None

    def __post_init__(self) -> None:
        """Validate record invariants."""

        if not self.symbol.strip():
            raise ValueError("symbol must be non-empty")


@dataclass(slots=True)
class DataBatch:
    """Collection of homogeneous records over a time interval.

    Args:
        records: Ordered records included in the batch.
        start_time: Inclusive start timestamp for the batch window.
        end_time: Inclusive end timestamp for the batch window.
        data_type: Data type represented by the batch.
        symbol: Symbol represented by the batch.
    """

    records: list[DataRecord]
    start_time: datetime
    end_time: datetime
    data_type: DataType
    symbol: str

    def __post_init__(self) -> None:
        """Validate batch shape and metadata consistency."""

        if not self.symbol.strip():
            raise ValueError("symbol must be non-empty")
        if self.start_time > self.end_time:
            raise ValueError("start_time must be less than or equal to end_time")

        for record in self.records:
            if record.symbol != self.symbol:
                raise ValueError("all records in batch must share the batch symbol")
            if record.data_type != self.data_type:
                raise ValueError("all records in batch must share the batch data_type")

    def to_dataframe(self) -> pd.DataFrame:
        """Convert records to a normalized dataframe.

        Returns:
            DataFrame with one row per record and flattened payload keys.
        """

        rows: list[dict[str, object]] = []
        for record in self.records:
            row: dict[str, object] = {
                "timestamp": record.timestamp,
                "data_type": record.data_type.value,
                "symbol": record.symbol,
                "source_id": record.source_id,
                "frequency": record.frequency.value if record.frequency else None,
            }
            row.update(record.payload)
            rows.append(row)

        return pd.DataFrame(rows)

    def filter_by_time(self, start: datetime, end: datetime) -> DataBatch:
        """Return a time-filtered view of the batch.

        Args:
            start: Inclusive start timestamp.
            end: Inclusive end timestamp.

        Returns:
            A new `DataBatch` constrained to the provided time window.
        """

        if start > end:
            raise ValueError("start must be less than or equal to end")

        filtered_records = [
            record for record in self.records if start <= record.timestamp <= end
        ]

        return DataBatch(
            records=filtered_records,
            start_time=max(self.start_time, start),
            end_time=min(self.end_time, end),
            data_type=self.data_type,
            symbol=self.symbol,
        )

    def __len__(self) -> int:
        """Return number of records in the batch."""

        return len(self.records)

    def __iter__(self) -> Iterator[DataRecord]:
        """Iterate over records in insertion order."""

        return iter(self.records)


@dataclass(frozen=True, slots=True)
class SourceMetadata:
    """Describes capabilities and configuration of a data source."""

    source_id: str
    source_type: str
    supported_types: list[DataType]
    supported_frequencies: list[DataFrequency]
    config: dict[str, object]


@dataclass(frozen=True, slots=True)
class NewsRecord:
    """Immutable representation of a news event for sentiment pipelines.

    Args:
        timestamp: Event publication time.
        headline: Headline text.
        body: Optional full body text.
        source: Provider/source label.
        symbols: Related tickers.
        categories: Topic/category tags.
        sentiment_score: Optional provider-supplied sentiment score.
        url: Optional article URL.
        news_id: Provider-unique event identifier.
    """

    timestamp: datetime
    headline: str
    body: str | None
    source: str
    symbols: list[str]
    categories: list[str]
    sentiment_score: float | None
    url: str | None
    news_id: str

    def __post_init__(self) -> None:
        """Validate core record constraints."""

        if not self.headline.strip():
            raise ValueError("headline must be non-empty")
        if not self.source.strip():
            raise ValueError("source must be non-empty")
        if not self.news_id.strip():
            raise ValueError("news_id must be non-empty")
        if self.sentiment_score is not None and not -1.0 <= self.sentiment_score <= 1.0:
            raise ValueError("sentiment_score must be between -1.0 and 1.0")


@dataclass(slots=True)
class NewsQuery:
    """Query parameters for historical/news API retrieval.

    Args:
        start_time: Inclusive query start timestamp.
        end_time: Inclusive query end timestamp.
        symbols: Optional symbol filters.
        keywords: Optional keyword filters. For providers without server-side
            text search support, filtering is applied client-side after fetch.
        categories: Optional category filters.
        limit: Maximum number of results returned.
        include_body: Whether full body text should be returned.
    """

    start_time: datetime
    end_time: datetime
    symbols: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    limit: int = 100
    include_body: bool = False

    def __post_init__(self) -> None:
        """Validate query constraints and normalize list values."""

        if self.start_time > self.end_time:
            raise ValueError("start_time must be less than or equal to end_time")
        if self.limit <= 0:
            raise ValueError("limit must be greater than 0")

        self.symbols = [value.strip() for value in self.symbols if value.strip()]
        self.keywords = [value.strip() for value in self.keywords if value.strip()]
        self.categories = [value.strip() for value in self.categories if value.strip()]
