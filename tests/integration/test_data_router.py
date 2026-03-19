"""Integration tests for DataRouter with concrete sources and targets."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from algotrading.src.data_pipeline.routing import (
    BufferConfig,
    ChannelConfig,
    DataRouter,
    MultiFrequencyBuffer,
    RouteConfig,
    RouterConfig,
)
from algotrading.src.data_pipeline.sources.file_source import FileSource
from algotrading.src.data_pipeline.state.state_manager import StateConfig, StateManager
from algotrading.src.data_pipeline.types import DataFrequency, DataRecord, DataType


def _wait_until(predicate: Callable[[], bool], timeout_seconds: float = 5.0) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _write_market_csv(base_path: Path, symbol: str, day: datetime) -> int:
    rows: list[dict[str, object]] = []
    for index in range(18):
        timestamp = day + timedelta(seconds=5 * index)
        value = 100.0 + index
        rows.append(
            {
                "date": timestamp.strftime("%Y%m%d %H:%M:%S") + " US/Eastern",
                "open": value - 0.2,
                "high": value + 0.3,
                "low": value - 0.4,
                "close": value,
                "volume": 1000 + index,
                "wap": value,
                "count": 10 + index,
            }
        )

    frame = pd.DataFrame(rows)
    filename = f"{symbol}_SMART_{day.strftime('%Y%m%d')}.csv"
    frame.to_csv(base_path / filename, index=False)
    return len(rows)


def _manager() -> StateManager:
    return StateManager(
        config=StateConfig(
            window_size=1,
            columns={
                "open": (False, True),
                "close": (False, True),
                "volume": (False, False),
            },
        )
    )


def test_router_single_source(tmp_path: Path) -> None:
    """Route a single file source into a state manager target."""

    row_count = _write_market_csv(tmp_path, "AAPL", datetime(2024, 1, 3, 9, 30))
    source = FileSource(base_path=str(tmp_path), stream_delay_seconds=0.0)
    manager = _manager()

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
    router.register_state_manager(manager)
    router.start()

    assert _wait_until(lambda: not router.is_streaming())
    assert len(manager.final_dataframe) == row_count


def test_router_multiple_sources(tmp_path: Path) -> None:
    """Route two sources simultaneously and receive both on callback target."""

    source_a_dir = tmp_path / "src_a"
    source_b_dir = tmp_path / "src_b"
    source_a_dir.mkdir(parents=True, exist_ok=True)
    source_b_dir.mkdir(parents=True, exist_ok=True)

    _write_market_csv(source_a_dir, "AAPL", datetime(2024, 1, 3, 9, 30))
    _write_market_csv(source_b_dir, "MSFT", datetime(2024, 1, 4, 9, 30))

    source_a = FileSource(base_path=str(source_a_dir), stream_delay_seconds=0.0)
    source_b = FileSource(base_path=str(source_b_dir), stream_delay_seconds=0.0)
    callback_records: list[DataRecord] = []

    router = DataRouter(
        RouterConfig(
            routes=[
                RouteConfig(
                    source=source_a,
                    data_types=[DataType.MARKET_BAR],
                    symbols=["AAPL"],
                    targets=["callback"],
                ),
                RouteConfig(
                    source=source_b,
                    data_types=[DataType.MARKET_BAR],
                    symbols=["MSFT"],
                    targets=["callback"],
                ),
            ]
        )
    )
    router.register_callback(callback_records.append, DataType.MARKET_BAR)
    router.start()

    assert _wait_until(lambda: not router.is_streaming())
    symbols = {record.symbol for record in callback_records}
    assert symbols == {"AAPL", "MSFT"}


def test_router_to_buffer(tmp_path: Path) -> None:
    """Route file source records into multi-frequency buffer."""

    _write_market_csv(tmp_path, "AAPL", datetime(2024, 1, 3, 9, 30))
    source = FileSource(base_path=str(tmp_path), stream_delay_seconds=0.0)

    buffer = MultiFrequencyBuffer(
        config=BufferConfig(
            channels=[
                ChannelConfig(
                    data_type=DataType.MARKET_BAR,
                    frequency=DataFrequency.SECOND_5,
                    max_size=64,
                )
            ]
        )
    )

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

    assert _wait_until(lambda: not router.is_streaming())
    latest = buffer.get_latest(DataType.MARKET_BAR)
    assert latest is not None
    assert latest.symbol == "AAPL"


def test_router_callbacks(tmp_path: Path) -> None:
    """Route records to custom callback handlers."""

    row_count = _write_market_csv(tmp_path, "AAPL", datetime(2024, 1, 3, 9, 30))
    source = FileSource(base_path=str(tmp_path), stream_delay_seconds=0.0)
    callback_records: list[DataRecord] = []

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
    router.register_callback(callback_records.append, DataType.MARKET_BAR)
    router.start()

    assert _wait_until(lambda: not router.is_streaming())
    assert len(callback_records) == row_count


def test_router_start_stop(tmp_path: Path) -> None:
    """Verify start/stop lifecycle management for the router."""

    _write_market_csv(tmp_path, "AAPL", datetime(2024, 1, 3, 9, 30))
    source = FileSource(base_path=str(tmp_path), stream_delay_seconds=0.005)
    manager = _manager()

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
    router.register_state_manager(manager)
    router.start()
    assert router.is_streaming()

    router.stop()
    assert not router.is_streaming()
