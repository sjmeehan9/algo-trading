"""End-to-end integration tests for the standalone data pipeline."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from algotrading.src.broker import BarData
from algotrading.src.data_pipeline.routing import (
    BufferConfig,
    ChannelConfig,
    DataRouter,
    MultiFrequencyBuffer,
    RouteConfig,
    RouterConfig,
)
from algotrading.src.data_pipeline.sources import BrokerDataSource, FileSource
from algotrading.src.data_pipeline.state import StateConfig, StateManager
from algotrading.src.data_pipeline.types import DataFrequency, DataRecord, DataType
from algotrading.src.data_sourcing.state_builder import StateBuilder

from tests.mocks import MockBrokerAdapter


def _wait_until(predicate: Callable[[], bool], timeout_seconds: float = 5.0) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _write_market_csv(path: Path, symbol: str = "AMD", rows: int = 40) -> Path:
    start = datetime(2024, 1, 5, 9, 30)
    payload: list[dict[str, object]] = []
    for index in range(rows):
        timestamp = start + timedelta(seconds=5 * index)
        close_price = 120.0 + index
        payload.append(
            {
                "date": timestamp.strftime("%Y%m%d %H:%M:%S") + " US/Eastern",
                "open": close_price - 0.2,
                "high": close_price + 0.3,
                "low": close_price - 0.4,
                "close": close_price,
                "volume": 1_000 + index,
                "wap": close_price,
                "count": 10 + index,
            }
        )

    file_path = path / f"{symbol}_SMART_20240105.csv"
    pd.DataFrame(payload).to_csv(file_path, index=False)
    return file_path


def _pipeline_state_manager(window_size: int = 10) -> StateManager:
    return StateManager(
        config=StateConfig(
            window_size=window_size,
            columns={
                "date": (True, False),
                "open": (False, True),
                "high": (False, True),
                "low": (False, True),
                "close": (False, True),
                "volume": (False, True),
                "wap": (False, True),
                "count": (False, True),
            },
            scaler_type="MinMaxScaler",
        )
    )


def _state_builder_config() -> tuple[dict, dict]:
    pipeline = {
        "pipeline": {
            "filename": "pipeline_0001",
            "contract_info": {"symbol": "AMD", "primaryExchange": "NASDAQ"},
            "state_data_config": {
                "columns": {
                    "date": [True, False],
                    "open": [False, True],
                    "high": [False, True],
                    "low": [False, True],
                    "close": [False, True],
                    "volume": [False, True],
                    "wap": [False, True],
                    "count": [False, True],
                },
                "file_trim": 0.0,
                "past_events": 10,
                "scaler": "MinMaxScaler",
            },
        }
    }
    config = {
        "task_selection": "task2",
        "data_mode": "historical",
        "data_path": "",
        "training_date_list": [],
        "backtest_date_list": [],
    }
    return pipeline, config


def test_full_pipeline_file_source(tmp_path: Path) -> None:
    """Run file source through router into state manager and build state."""

    _write_market_csv(tmp_path, rows=40)
    manager = _pipeline_state_manager(window_size=10)
    source = FileSource(
        base_path=str(tmp_path), stream_delay_seconds=0.0, trim_percentage=0.0
    )

    router = DataRouter(
        RouterConfig(
            routes=[
                RouteConfig(
                    source=source,
                    data_types=[DataType.MARKET_BAR],
                    symbols=["AMD"],
                    targets=["state_manager"],
                )
            ]
        )
    )
    router.register_state_manager(manager)
    router.start()
    assert _wait_until(lambda: not router.is_streaming())

    manager.episode_length = len(manager.final_dataframe) - manager.config.window_size
    manager.total_timesteps = manager.episode_length
    state = manager.initialise_state()

    assert len(manager.final_dataframe) == 40
    assert set(state.keys()) == {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "wap",
        "count",
    }


def test_full_pipeline_broker_source() -> None:
    """Run broker source through router into state manager using mock adapter."""

    adapter = MockBrokerAdapter()
    adapter.connect("127.0.0.1", 7497, 111)
    source = BrokerDataSource(
        adapter=adapter, default_symbol="AMD", stream_queue_timeout_seconds=0.1
    )
    manager = StateManager(
        config=StateConfig(
            window_size=5,
            columns={
                "date": (True, False),
                "open": (False, True),
                "high": (False, True),
                "low": (False, True),
                "close": (False, True),
                "volume": (False, True),
                "vwap": (False, True),
                "trade_count": (False, True),
            },
            scaler_type="MinMaxScaler",
        )
    )

    router = DataRouter(
        RouterConfig(
            routes=[
                RouteConfig(
                    source=source,
                    data_types=[DataType.MARKET_BAR],
                    symbols=["AMD"],
                    targets=["state_manager"],
                )
            ]
        )
    )
    router.register_state_manager(manager)
    router.start()

    base_time = datetime(2024, 1, 5, 14, 30, tzinfo=UTC)
    for index in range(8):
        close_price = 150.0 + index
        adapter.emit_realtime_bar(
            BarData(
                timestamp=base_time + timedelta(seconds=5 * index),
                open=close_price - 0.2,
                high=close_price + 0.3,
                low=close_price - 0.4,
                close=close_price,
                volume=2_000 + index,
                vwap=close_price,
                trade_count=20 + index,
            )
        )

    assert _wait_until(lambda: len(manager.final_dataframe) >= 8)
    router.stop()
    adapter.disconnect()

    manager.episode_length = len(manager.final_dataframe) - manager.config.window_size
    manager.total_timesteps = manager.episode_length
    state = manager.initialise_state()

    assert len(manager.final_dataframe) >= 8
    assert "close" in state


def test_multi_frequency_alignment() -> None:
    """Align market bars and irregular news records by timestamp."""

    buffer = MultiFrequencyBuffer(
        config=BufferConfig(
            channels=[
                ChannelConfig(
                    data_type=DataType.MARKET_BAR,
                    frequency=DataFrequency.SECOND_5,
                    max_size=64,
                ),
                ChannelConfig(
                    data_type=DataType.NEWS_TEXT,
                    frequency=DataFrequency.IRREGULAR,
                    max_size=64,
                ),
            ]
        )
    )

    t0 = datetime(2024, 1, 5, 9, 30, tzinfo=UTC)
    buffer.put(
        DataRecord(
            timestamp=t0,
            data_type=DataType.MARKET_BAR,
            symbol="AMD",
            payload={"close": 100.0},
            source_id="market",
            frequency=DataFrequency.SECOND_5,
        )
    )
    buffer.put(
        DataRecord(
            timestamp=t0 + timedelta(seconds=8),
            data_type=DataType.NEWS_TEXT,
            symbol="AMD",
            payload={"headline": "AMD announces guidance", "sentiment": 0.7},
            source_id="news",
            frequency=DataFrequency.IRREGULAR,
        )
    )
    buffer.put(
        DataRecord(
            timestamp=t0 + timedelta(seconds=10),
            data_type=DataType.MARKET_BAR,
            symbol="AMD",
            payload={"close": 101.0},
            source_id="market",
            frequency=DataFrequency.SECOND_5,
        )
    )

    aligned = buffer.get_aligned_state(t0 + timedelta(seconds=12))
    assert DataType.MARKET_BAR in aligned
    assert DataType.NEWS_TEXT in aligned
    assert aligned[DataType.MARKET_BAR].payload["close"] == 101.0
    assert aligned[DataType.NEWS_TEXT].payload["headline"] == "AMD announces guidance"


def test_pipeline_training_comparison(tmp_path: Path) -> None:
    """Compare training-state inputs between StateBuilder and pipeline path."""

    file_path = _write_market_csv(tmp_path, symbol="AMD", rows=50)
    dataframe = pd.read_csv(file_path)

    pipeline, config = _state_builder_config()
    legacy = StateBuilder(
        config,
        pipeline,
        custom_logic=type(
            "_Noop",
            (),
            {
                "initialise_variables": staticmethod(lambda: {}),
                "step": staticmethod(
                    lambda action, state_df, custom_variable_dict, terminated: {}
                ),
            },
        )(),
    )
    legacy.final_dataframe = dataframe
    legacy.episode_length = 40
    legacy.total_timesteps = 40
    legacy.initialise_state()

    source = FileSource(base_path=str(tmp_path), trim_percentage=0.0)
    source.connect()
    batch = source.fetch_batch(
        symbol="AMD",
        start=datetime(2024, 1, 5, 0, 0, tzinfo=ZoneInfo("US/Eastern")),
        end=datetime(2024, 1, 5, 23, 59, tzinfo=ZoneInfo("US/Eastern")),
    )

    manager = _pipeline_state_manager(window_size=10)
    manager.load_data(batch)
    manager.episode_length = 40
    manager.total_timesteps = 40
    manager.initialise_state()

    assert set(manager.state.keys()) == set(legacy.state.keys())
    for key in legacy.state:
        np.testing.assert_allclose(manager.state[key], legacy.state[key], atol=1e-9)


def test_pipeline_backtest_comparison(tmp_path: Path) -> None:
    """Compare backtest-state stepping between StateBuilder and pipeline path."""

    file_path = _write_market_csv(tmp_path, symbol="AMD", rows=55)
    dataframe = pd.read_csv(file_path)

    pipeline, config = _state_builder_config()
    config["task_selection"] = "task4"
    noop_logic = type(
        "_Noop",
        (),
        {
            "initialise_variables": staticmethod(lambda: {}),
            "step": staticmethod(
                lambda action, state_df, custom_variable_dict, terminated: {}
            ),
        },
    )()

    legacy = StateBuilder(config, pipeline, custom_logic=noop_logic)
    legacy.final_dataframe = dataframe
    legacy.episode_length = 45
    legacy.total_timesteps = 45
    legacy.initialise_state()
    legacy.state_step(1)

    source = FileSource(base_path=str(tmp_path), trim_percentage=0.0)
    source.connect()
    batch = source.fetch_batch(
        symbol="AMD",
        start=datetime(2024, 1, 5, 0, 0, tzinfo=ZoneInfo("US/Eastern")),
        end=datetime(2024, 1, 5, 23, 59, tzinfo=ZoneInfo("US/Eastern")),
    )
    manager = _pipeline_state_manager(window_size=10)
    manager.custom_logic = noop_logic
    manager.load_data(batch)
    manager.episode_length = 45
    manager.total_timesteps = 45
    manager.initialise_state()
    manager.state_step(1)

    assert set(manager.state.keys()) == set(legacy.state.keys())
    for key in legacy.state:
        np.testing.assert_allclose(manager.state[key], legacy.state[key], atol=1e-9)


def test_pipeline_memory_usage() -> None:
    """Verify configured buffer memory bounds are respected."""

    buffer = MultiFrequencyBuffer(
        config=BufferConfig(
            channels=[
                ChannelConfig(
                    data_type=DataType.MARKET_BAR,
                    frequency=DataFrequency.SECOND_5,
                    max_size=20,
                )
            ]
        )
    )

    start = datetime(2024, 1, 5, 9, 30, tzinfo=UTC)
    for index in range(200):
        buffer.put(
            DataRecord(
                timestamp=start + timedelta(seconds=5 * index),
                data_type=DataType.MARKET_BAR,
                symbol="AMD",
                payload={"close": float(100 + index)},
                source_id="memory-test",
                frequency=DataFrequency.SECOND_5,
            )
        )

    usage = buffer.get_memory_usage()
    assert usage[DataType.MARKET_BAR] <= 20
