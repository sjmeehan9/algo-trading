"""Integration tests for multi-frequency buffer alignment across stream types."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from algotrading.src.data_pipeline.routing import (
    BufferConfig,
    ChannelConfig,
    MultiFrequencyBuffer,
)
from algotrading.src.data_pipeline.types import DataFrequency, DataRecord, DataType


def _market_record(timestamp: datetime, close: float) -> DataRecord:
    return DataRecord(
        timestamp=timestamp,
        data_type=DataType.MARKET_BAR,
        symbol="AMD",
        payload={"close": close},
        source_id="integration-market",
        frequency=DataFrequency.SECOND_5,
    )


def _news_record(timestamp: datetime, sentiment: float) -> DataRecord:
    return DataRecord(
        timestamp=timestamp,
        data_type=DataType.NEWS_TEXT,
        symbol="AMD",
        payload={"sentiment": sentiment},
        source_id="integration-news",
        frequency=DataFrequency.IRREGULAR,
    )


def test_multi_frequency_buffer_aligns_market_and_news_with_gaps() -> None:
    """Alignment returns latest available slower-frequency values at query times."""

    base = datetime(2026, 2, 25, 14, 30, tzinfo=UTC)
    buffer = MultiFrequencyBuffer(
        BufferConfig(
            channels=[
                ChannelConfig(
                    DataType.MARKET_BAR, DataFrequency.SECOND_5, max_size=500
                ),
                ChannelConfig(
                    DataType.NEWS_TEXT, DataFrequency.IRREGULAR, max_size=100
                ),
            ]
        )
    )

    for index in range(10):
        buffer.put(
            _market_record(
                timestamp=base + timedelta(seconds=index * 5),
                close=100.0 + index,
            )
        )

    buffer.put(_news_record(base + timedelta(seconds=7), sentiment=0.2))
    buffer.put(_news_record(base + timedelta(seconds=31), sentiment=-0.4))

    aligned_1 = buffer.get_aligned_state(base + timedelta(seconds=12))
    assert aligned_1[DataType.MARKET_BAR].payload["close"] == 102.0
    assert aligned_1[DataType.NEWS_TEXT].payload["sentiment"] == 0.2

    aligned_2 = buffer.get_aligned_state(base + timedelta(seconds=29))
    assert aligned_2[DataType.MARKET_BAR].payload["close"] == 105.0
    assert aligned_2[DataType.NEWS_TEXT].payload["sentiment"] == 0.2

    aligned_3 = buffer.get_aligned_state(base + timedelta(seconds=33))
    assert aligned_3[DataType.MARKET_BAR].payload["close"] == 106.0
    assert aligned_3[DataType.NEWS_TEXT].payload["sentiment"] == -0.4
