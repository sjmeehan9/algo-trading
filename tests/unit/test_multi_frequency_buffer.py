"""Unit tests for multi-frequency data buffering and timestamp alignment."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

from algotrading.src.data_pipeline.routing import (
    BufferChannel,
    BufferConfig,
    ChannelConfig,
    MultiFrequencyBuffer,
)
from algotrading.src.data_pipeline.types import DataFrequency, DataRecord, DataType


def _record(
    timestamp: datetime,
    data_type: DataType,
    frequency: DataFrequency,
    value: float,
    symbol: str = "AMD",
) -> DataRecord:
    return DataRecord(
        timestamp=timestamp,
        data_type=data_type,
        symbol=symbol,
        payload={"value": value},
        source_id="unit-test",
        frequency=frequency,
    )


def test_buffer_channel_put_preserves_order_and_get_latest() -> None:
    """BufferChannel maintains timestamp ordering and latest<=time lookup."""

    base = datetime(2026, 2, 25, 14, 0, tzinfo=UTC)
    channel = BufferChannel(
        ChannelConfig(
            data_type=DataType.MARKET_BAR,
            frequency=DataFrequency.SECOND_5,
            max_size=10,
        )
    )

    channel.put(
        _record(
            base + timedelta(seconds=10),
            DataType.MARKET_BAR,
            DataFrequency.SECOND_5,
            1.0,
        )
    )
    channel.put(
        _record(
            base + timedelta(seconds=5),
            DataType.MARKET_BAR,
            DataFrequency.SECOND_5,
            2.0,
        )
    )
    channel.put(
        _record(
            base + timedelta(seconds=15),
            DataType.MARKET_BAR,
            DataFrequency.SECOND_5,
            3.0,
        )
    )

    records = channel.get_range(base, base + timedelta(seconds=20))
    assert [record.timestamp for record in records] == [
        base + timedelta(seconds=5),
        base + timedelta(seconds=10),
        base + timedelta(seconds=15),
    ]
    assert channel.get_latest(before=base + timedelta(seconds=12)) == records[1]
    assert channel.get_latest(before=base + timedelta(seconds=3)) is None


def test_buffer_channel_enforces_max_size() -> None:
    """BufferChannel evicts oldest records when max size is exceeded."""

    base = datetime(2026, 2, 25, 14, 0, tzinfo=UTC)
    channel = BufferChannel(
        ChannelConfig(
            data_type=DataType.MARKET_BAR,
            frequency=DataFrequency.SECOND_5,
            max_size=3,
        )
    )

    for index in range(5):
        channel.put(
            _record(
                base + timedelta(seconds=index),
                DataType.MARKET_BAR,
                DataFrequency.SECOND_5,
                float(index),
            )
        )

    retained = channel.get_range(base, base + timedelta(seconds=10))
    assert len(channel) == 3
    assert [record.payload["value"] for record in retained] == [2.0, 3.0, 4.0]


def test_buffer_channel_ttl_prunes_old_records() -> None:
    """BufferChannel removes records older than configured TTL horizon."""

    base = datetime(2026, 2, 25, 14, 0, tzinfo=UTC)
    channel = BufferChannel(
        ChannelConfig(
            data_type=DataType.MARKET_BAR,
            frequency=DataFrequency.SECOND_5,
            max_size=10,
            ttl_seconds=10,
        )
    )

    channel.put(_record(base, DataType.MARKET_BAR, DataFrequency.SECOND_5, 1.0))
    channel.put(
        _record(
            base + timedelta(seconds=5),
            DataType.MARKET_BAR,
            DataFrequency.SECOND_5,
            2.0,
        )
    )
    channel.put(
        _record(
            base + timedelta(seconds=20),
            DataType.MARKET_BAR,
            DataFrequency.SECOND_5,
            3.0,
        )
    )

    retained = channel.get_range(base, base + timedelta(seconds=25))
    assert len(retained) == 1
    assert retained[0].payload["value"] == 3.0


def test_multi_frequency_buffer_routes_and_aligns_records() -> None:
    """MultiFrequencyBuffer aligns latest records across data types by time."""

    base = datetime(2026, 2, 25, 14, 0, tzinfo=UTC)
    buffer = MultiFrequencyBuffer(
        BufferConfig(
            channels=[
                ChannelConfig(DataType.MARKET_BAR, DataFrequency.SECOND_5),
                ChannelConfig(DataType.NEWS_TEXT, DataFrequency.IRREGULAR),
            ]
        )
    )

    buffer.put(
        _record(
            base + timedelta(seconds=10),
            DataType.MARKET_BAR,
            DataFrequency.SECOND_5,
            101.0,
        )
    )
    buffer.put(
        _record(
            base + timedelta(seconds=15),
            DataType.MARKET_BAR,
            DataFrequency.SECOND_5,
            102.0,
        )
    )
    buffer.put(
        _record(
            base + timedelta(seconds=12),
            DataType.NEWS_TEXT,
            DataFrequency.IRREGULAR,
            0.5,
        )
    )

    aligned = buffer.get_aligned_state(base + timedelta(seconds=14))
    assert aligned[DataType.MARKET_BAR].payload["value"] == 101.0
    assert aligned[DataType.NEWS_TEXT].payload["value"] == 0.5

    filtered = buffer.get_aligned_state(
        base + timedelta(seconds=14),
        channels=[DataType.NEWS_TEXT],
    )
    assert set(filtered.keys()) == {DataType.NEWS_TEXT}


def test_multi_frequency_buffer_get_latest_uses_newest_across_frequencies() -> None:
    """Latest lookup returns newest record among channels sharing a data type."""

    base = datetime(2026, 2, 25, 14, 0, tzinfo=UTC)
    buffer = MultiFrequencyBuffer(
        BufferConfig(
            channels=[
                ChannelConfig(DataType.MARKET_BAR, DataFrequency.SECOND_5),
                ChannelConfig(DataType.MARKET_BAR, DataFrequency.MINUTE_1),
            ]
        )
    )

    buffer.put(
        _record(
            base + timedelta(seconds=30),
            DataType.MARKET_BAR,
            DataFrequency.SECOND_5,
            101.0,
        )
    )
    buffer.put(
        _record(
            base + timedelta(seconds=60),
            DataType.MARKET_BAR,
            DataFrequency.MINUTE_1,
            103.0,
        )
    )

    latest = buffer.get_latest(DataType.MARKET_BAR)
    assert latest is not None
    assert latest.payload["value"] == 103.0


def test_multi_frequency_buffer_prune_and_memory_usage() -> None:
    """Prune and memory usage report expected counts by data type."""

    base = datetime(2026, 2, 25, 14, 0, tzinfo=UTC)
    buffer = MultiFrequencyBuffer(
        BufferConfig(
            channels=[
                ChannelConfig(DataType.MARKET_BAR, DataFrequency.SECOND_5),
                ChannelConfig(DataType.NEWS_TEXT, DataFrequency.IRREGULAR),
            ]
        )
    )

    buffer.put(
        _record(
            base + timedelta(seconds=0),
            DataType.MARKET_BAR,
            DataFrequency.SECOND_5,
            100.0,
        )
    )
    buffer.put(
        _record(
            base + timedelta(seconds=10),
            DataType.MARKET_BAR,
            DataFrequency.SECOND_5,
            101.0,
        )
    )
    buffer.put(
        _record(
            base + timedelta(seconds=12),
            DataType.NEWS_TEXT,
            DataFrequency.IRREGULAR,
            0.2,
        )
    )

    removed = buffer.prune_all(base + timedelta(seconds=10))
    assert removed == 1

    usage = buffer.get_memory_usage()
    assert usage[DataType.MARKET_BAR] == 1
    assert usage[DataType.NEWS_TEXT] == 1


def test_multi_frequency_buffer_is_thread_safe_for_concurrent_read_write() -> None:
    """Concurrent writers/readers execute without race errors."""

    base = datetime(2026, 2, 25, 14, 0, tzinfo=UTC)
    buffer = MultiFrequencyBuffer(
        BufferConfig(
            channels=[
                ChannelConfig(
                    DataType.MARKET_BAR, DataFrequency.SECOND_1, max_size=10000
                )
            ]
        )
    )

    def _writer(start_index: int, count: int) -> None:
        for offset in range(count):
            sequence = start_index + offset
            buffer.put(
                _record(
                    base + timedelta(seconds=sequence),
                    DataType.MARKET_BAR,
                    DataFrequency.SECOND_1,
                    float(sequence),
                )
            )

    def _reader(iterations: int) -> None:
        for _ in range(iterations):
            _ = buffer.get_latest(DataType.MARKET_BAR)
            _ = buffer.get_aligned_state(base + timedelta(hours=1))

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [
            executor.submit(_writer, 0, 200),
            executor.submit(_writer, 200, 200),
            executor.submit(_writer, 400, 200),
            executor.submit(_reader, 300),
            executor.submit(_reader, 300),
            executor.submit(_reader, 300),
        ]
        for future in futures:
            future.result()

    usage = buffer.get_memory_usage()
    assert usage[DataType.MARKET_BAR] == 600
    latest = buffer.get_latest(DataType.MARKET_BAR)
    assert latest is not None
    assert latest.timestamp == base + timedelta(seconds=599)
