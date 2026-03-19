"""Unit tests for streaming data router behavior."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Iterator

from algotrading.src.data_pipeline.routing import (
    BufferConfig,
    ChannelConfig,
    DataRouter,
    MultiFrequencyBuffer,
    RouteConfig,
    RouterConfig,
)
from algotrading.src.data_pipeline.sources.base import DataSource
from algotrading.src.data_pipeline.state.state_manager import StateConfig, StateManager
from algotrading.src.data_pipeline.types import (
    DataBatch,
    DataFrequency,
    DataRecord,
    DataType,
    SourceMetadata,
)


@dataclass(frozen=True, slots=True)
class _SourceKey:
    """Lookup key for test source stream payloads."""

    symbol: str
    data_type: DataType


class _InMemorySource(DataSource):
    """In-memory source implementation for deterministic router tests."""

    def __init__(
        self,
        source_name: str,
        stream_data: dict[_SourceKey, list[DataRecord]],
        *,
        delay_seconds: float = 0.0,
        fail_first_stream: bool = False,
    ) -> None:
        self._source_name = source_name
        self._stream_data = stream_data
        self._delay_seconds = delay_seconds
        self._connected = False
        self._fail_first_stream = fail_first_stream
        self._stream_calls = 0

    @property
    def source_id(self) -> str:
        return self._source_name

    @property
    def metadata(self) -> SourceMetadata:
        supported_types = sorted(
            {key.data_type for key in self._stream_data.keys()},
            key=lambda item: item.value,
        )
        return SourceMetadata(
            source_id=self.source_id,
            source_type="memory",
            supported_types=supported_types,
            supported_frequencies=[DataFrequency.SECOND_5, DataFrequency.IRREGULAR],
            config={},
        )

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def fetch_batch(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        data_type: DataType = DataType.MARKET_BAR,
    ) -> DataBatch:
        records = [
            record
            for record in self._stream_data.get(_SourceKey(symbol, data_type), [])
            if start <= record.timestamp <= end
        ]
        if records:
            batch_start = records[0].timestamp
            batch_end = records[-1].timestamp
        else:
            batch_start = start
            batch_end = end
        return DataBatch(
            records=records,
            start_time=batch_start,
            end_time=batch_end,
            data_type=data_type,
            symbol=symbol,
        )

    def fetch_stream(
        self,
        symbol: str,
        data_type: DataType = DataType.MARKET_BAR,
    ) -> Iterator[DataRecord]:
        self._stream_calls += 1
        if self._fail_first_stream and self._stream_calls == 1:
            raise RuntimeError("simulated stream failure")

        key = _SourceKey(symbol, data_type)
        for record in self._stream_data.get(key, []):
            if self._delay_seconds > 0:
                time.sleep(self._delay_seconds)
            yield record

    def get_available_symbols(self) -> list[str]:
        return sorted({key.symbol for key in self._stream_data.keys()})

    def get_available_dates(self, symbol: str) -> list[date]:
        dates = {
            record.timestamp.date()
            for key, records in self._stream_data.items()
            if key.symbol == symbol
            for record in records
        }
        return sorted(dates)


def _make_record(
    *,
    timestamp: datetime,
    symbol: str = "AAPL",
    data_type: DataType = DataType.MARKET_BAR,
    frequency: DataFrequency = DataFrequency.SECOND_5,
    close_value: float = 100.0,
) -> DataRecord:
    payload = {
        "open": close_value - 0.5,
        "high": close_value + 0.5,
        "low": close_value - 1.0,
        "close": close_value,
        "volume": 1000,
        "wap": close_value,
        "count": 10,
    }
    return DataRecord(
        timestamp=timestamp,
        data_type=data_type,
        symbol=symbol,
        payload=payload,
        source_id="test-source",
        frequency=frequency,
    )


def _wait_until(predicate: Callable[[], bool], timeout_seconds: float = 2.0) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _state_manager() -> StateManager:
    config = StateConfig(
        window_size=1,
        columns={
            "open": (False, True),
            "close": (False, True),
            "volume": (False, False),
        },
    )
    return StateManager(config=config)


def test_handler_registration_and_unregistration() -> None:
    router = DataRouter()

    def _handler(record: DataRecord) -> None:
        del record

    handler_id = router.register_callback(_handler, DataType.MARKET_BAR)
    router.unregister_callback(handler_id)

    assert handler_id.startswith("handler_")


def test_source_addition_and_removal() -> None:
    now = datetime.now(UTC)
    source = _InMemorySource(
        "src_add_remove",
        {
            _SourceKey("AAPL", DataType.MARKET_BAR): [_make_record(timestamp=now)],
        },
    )
    route = RouteConfig(
        source=source,
        data_types=[DataType.MARKET_BAR],
        symbols=["AAPL"],
        targets=["callback"],
    )

    router = DataRouter()
    router.add_source(source, route)
    assert len(router.get_sources()) == 1

    router.remove_source(source.source_id)
    assert len(router.get_sources()) == 0


def test_router_routes_to_state_manager() -> None:
    now = datetime.now(UTC)
    source = _InMemorySource(
        "src_state_manager",
        {
            _SourceKey("AAPL", DataType.MARKET_BAR): [
                _make_record(timestamp=now, close_value=101.0),
                _make_record(timestamp=now + timedelta(seconds=5), close_value=102.0),
            ]
        },
    )

    router = DataRouter(
        RouterConfig(
            routes=[
                RouteConfig(
                    source=source,
                    data_types=[DataType.MARKET_BAR],
                    symbols=["AAPL"],
                    targets=["state_manager"],
                )
            ]
        )
    )
    manager = _state_manager()
    router.register_state_manager(manager)

    router.start()
    assert _wait_until(lambda: not router.is_streaming(), timeout_seconds=2.0)

    assert len(manager.final_dataframe) == 2
    assert manager.final_dataframe.iloc[-1]["close"] == 102.0


def test_router_routes_to_buffer() -> None:
    now = datetime.now(UTC)
    source = _InMemorySource(
        "src_buffer",
        {
            _SourceKey("AAPL", DataType.MARKET_BAR): [
                _make_record(timestamp=now, close_value=201.0),
                _make_record(timestamp=now + timedelta(seconds=5), close_value=202.0),
            ]
        },
    )

    buffer_config = BufferConfig(
        channels=[
            ChannelConfig(
                data_type=DataType.MARKET_BAR,
                frequency=DataFrequency.SECOND_5,
                max_size=10,
            )
        ]
    )

    buffer = MultiFrequencyBuffer(config=buffer_config)

    router = DataRouter(
        RouterConfig(
            routes=[
                RouteConfig(
                    source=source,
                    data_types=[DataType.MARKET_BAR],
                    symbols=["AAPL"],
                    targets=["buffer"],
                )
            ]
        )
    )
    router.register_buffer(buffer)

    router.start()
    assert _wait_until(lambda: not router.is_streaming(), timeout_seconds=2.0)

    latest = buffer.get_latest(DataType.MARKET_BAR)
    assert latest is not None
    assert latest.payload["close"] == 202.0


def test_router_routes_to_callbacks() -> None:
    now = datetime.now(UTC)
    source = _InMemorySource(
        "src_callback",
        {
            _SourceKey("AAPL", DataType.MARKET_BAR): [
                _make_record(timestamp=now, close_value=301.0),
                _make_record(timestamp=now + timedelta(seconds=5), close_value=302.0),
            ]
        },
    )

    captured: list[DataRecord] = []
    router = DataRouter(
        RouterConfig(
            routes=[
                RouteConfig(
                    source=source,
                    data_types=[DataType.MARKET_BAR],
                    symbols=["AAPL"],
                    targets=["callback"],
                )
            ]
        )
    )
    router.register_callback(captured.append, DataType.MARKET_BAR)

    router.start()
    assert _wait_until(lambda: not router.is_streaming(), timeout_seconds=2.0)

    assert [record.payload["close"] for record in captured] == [301.0, 302.0]


def test_start_and_stop_streaming_controls_data_flow() -> None:
    now = datetime.now(UTC)
    records = [
        _make_record(
            timestamp=now + timedelta(seconds=index * 5), close_value=400 + index
        )
        for index in range(30)
    ]
    source = _InMemorySource(
        "src_start_stop",
        {_SourceKey("AAPL", DataType.MARKET_BAR): records},
        delay_seconds=0.02,
    )

    captured: list[DataRecord] = []
    router = DataRouter(
        RouterConfig(
            routes=[
                RouteConfig(
                    source=source,
                    data_types=[DataType.MARKET_BAR],
                    symbols=["AAPL"],
                    targets=["callback"],
                )
            ]
        )
    )
    router.register_callback(captured.append, DataType.MARKET_BAR)

    router.start_streaming(source_id=source.source_id)
    assert router.is_streaming(source.source_id)

    time.sleep(0.12)
    router.stop_streaming(source_id=source.source_id)

    assert not router.is_streaming(source.source_id)
    assert 0 < len(captured) < len(records)


def test_router_calls_error_callbacks_on_source_failure() -> None:
    now = datetime.now(UTC)
    source = _InMemorySource(
        "src_fail",
        {
            _SourceKey("AAPL", DataType.MARKET_BAR): [
                _make_record(timestamp=now, close_value=501.0)
            ]
        },
        fail_first_stream=True,
    )

    errors: list[tuple[str, str]] = []
    router = DataRouter(
        RouterConfig(
            routes=[
                RouteConfig(
                    source=source,
                    data_types=[DataType.MARKET_BAR],
                    symbols=["AAPL"],
                    targets=["callback"],
                )
            ],
            retry_attempts=0,
        )
    )
    router.register_error_callback(
        lambda source_id, error: errors.append((source_id, str(error)))
    )

    router.start()
    assert _wait_until(lambda: not router.is_streaming(), timeout_seconds=2.0)

    assert len(errors) == 1
    assert errors[0][0] == "src_fail"
    assert "simulated stream failure" in errors[0][1]
