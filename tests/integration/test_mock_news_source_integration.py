"""Integration tests for mock news source with router and multi-frequency buffer."""

from __future__ import annotations

import time
from collections.abc import Callable

from algotrading.src.data_pipeline.routing import (
    BufferConfig,
    ChannelConfig,
    DataRouter,
    MultiFrequencyBuffer,
    RouteConfig,
    RouterConfig,
)
from algotrading.src.data_pipeline.sources.mock_news_source import MockNewsSource
from algotrading.src.data_pipeline.types import DataFrequency, DataRecord, DataType


def _wait_until(predicate: Callable[[], bool], timeout_seconds: float = 3.0) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_mock_news_source_streams_through_router_into_buffer() -> None:
    """Router ingests mock news stream and writes aligned records to buffer."""

    source = MockNewsSource(symbols=["AAPL"], frequency_seconds=1)
    buffer = MultiFrequencyBuffer(
        config=BufferConfig(
            channels=[
                ChannelConfig(
                    data_type=DataType.NEWS_TEXT,
                    frequency=DataFrequency.IRREGULAR,
                    max_size=32,
                )
            ]
        )
    )

    callback_records: list[DataRecord] = []

    router = DataRouter(
        RouterConfig(
            routes=[
                RouteConfig(
                    source=source,
                    data_types=[DataType.NEWS_TEXT],
                    symbols=["AAPL"],
                    targets=["buffer", "callback"],
                )
            ]
        )
    )
    router.register_buffer(buffer)
    router.register_callback(callback_records.append, DataType.NEWS_TEXT)

    with router:
        assert _wait_until(lambda: len(callback_records) >= 1)

    latest = buffer.get_latest(DataType.NEWS_TEXT)
    assert latest is not None
    assert latest.symbol == "AAPL"
    assert "headline" in latest.payload
    assert len(callback_records) >= 1
