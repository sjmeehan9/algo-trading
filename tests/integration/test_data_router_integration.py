"""Integration tests for data router with concrete pipeline components."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime, timedelta

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


def _write_market_csv(base_path: str) -> int:
    start = datetime(2024, 1, 1, 9, 30)
    rows: list[dict[str, object]] = []
    for index in range(20):
        timestamp = start + timedelta(seconds=5 * index)
        value = 100.0 + index
        rows.append(
            {
                "date": timestamp.strftime("%Y%m%d %H:%M:%S") + " US/Eastern",
                "open": value - 0.1,
                "high": value + 0.2,
                "low": value - 0.4,
                "close": value,
                "volume": 1000 + index,
                "wap": value,
                "count": 10 + index,
            }
        )

    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(f"{base_path}/AAPL_SMART_20240101.csv", index=False)
    return len(rows)


def _build_state_manager() -> StateManager:
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


def test_router_with_file_source_stream_mode(tmp_path) -> None:
    row_count = _write_market_csv(str(tmp_path))

    source = FileSource(base_path=str(tmp_path), stream_delay_seconds=0.0)
    manager = _build_state_manager()
    buffer = MultiFrequencyBuffer(
        config=BufferConfig(
            channels=[
                ChannelConfig(
                    data_type=DataType.MARKET_BAR,
                    frequency=DataFrequency.SECOND_5,
                    max_size=128,
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
                    data_types=[DataType.MARKET_BAR],
                    symbols=["AAPL"],
                    targets=["state_manager", "buffer", "callback"],
                )
            ]
        )
    )
    router.register_state_manager(manager, DataType.MARKET_BAR)
    router.register_buffer(buffer)
    router.register_callback(callback_records.append, DataType.MARKET_BAR)

    router.start()
    assert _wait_until(lambda: not router.is_streaming(), timeout_seconds=5.0)

    assert len(callback_records) == row_count
    assert len(manager.final_dataframe) == row_count

    latest = buffer.get_latest(DataType.MARKET_BAR)
    assert latest is not None
    assert latest.symbol == "AAPL"
    assert latest.payload["close"] == 119.0


def test_router_feeds_state_manager_from_file_source(tmp_path) -> None:
    row_count = _write_market_csv(str(tmp_path))

    source = FileSource(base_path=str(tmp_path), stream_delay_seconds=0.0)
    manager = _build_state_manager()

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
    router.register_state_manager(manager, DataType.MARKET_BAR)

    with router:
        assert _wait_until(lambda: not router.is_streaming(), timeout_seconds=5.0)

    assert len(manager.final_dataframe) == row_count
    assert manager.final_dataframe.iloc[0]["close"] == 100.0
    assert manager.final_dataframe.iloc[-1]["close"] == 119.0
