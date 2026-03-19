"""Integration tests for StateManager end-to-end behavior."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from algotrading.src.data_pipeline.state import StateConfig, StateManager
from algotrading.src.data_pipeline.types import (
    DataBatch,
    DataFrequency,
    DataRecord,
    DataType,
)
from algotrading.src.data_sourcing.state_builder import StateBuilder


@dataclass(slots=True)
class _CustomLogicStub:
    """Custom logic stub for validating custom variable integration."""

    def initialise_variables(self) -> dict[str, float]:
        return {"current_position": 0.0, "trade_change": 0.0}

    def step(
        self,
        action: int,
        state_df: pd.DataFrame,
        custom_variable_dict: dict[str, np.ndarray],
        terminated: bool,
    ) -> dict[str, np.ndarray]:
        del state_df, terminated
        previous_position = custom_variable_dict.get("current_position", np.array([]))
        previous_change = custom_variable_dict.get("trade_change", np.array([]))
        return {
            "current_position": np.append(previous_position, float(action)),
            "trade_change": np.append(previous_change, float(action != 0)),
        }


def _build_market_records(count: int = 50) -> list[DataRecord]:
    start = datetime(2024, 1, 3, 9, 30)
    records: list[DataRecord] = []
    for index in range(count):
        timestamp = start + timedelta(seconds=5 * index)
        price = 100.0 + index
        records.append(
            DataRecord(
                timestamp=timestamp,
                data_type=DataType.MARKET_BAR,
                symbol="AMD",
                payload={
                    "open": price - 0.1,
                    "high": price + 0.2,
                    "low": price - 0.3,
                    "close": price,
                    "volume": 1000 + index,
                    "wap": price + 0.05,
                    "count": 10 + index,
                },
                source_id="integration-source",
                frequency=DataFrequency.SECOND_5,
            )
        )
    return records


def _build_batch(record_count: int = 50) -> DataBatch:
    records = _build_market_records(record_count)
    return DataBatch(
        records=records,
        start_time=records[0].timestamp,
        end_time=records[-1].timestamp,
        data_type=DataType.MARKET_BAR,
        symbol="AMD",
    )


def _build_state_config(window_size: int = 10) -> StateConfig:
    return StateConfig(
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


def _build_pipeline_and_config() -> tuple[dict[str, Any], dict[str, Any]]:
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


def test_state_manager_initialization() -> None:
    """Create manager and verify it initializes with expected counters."""

    manager = StateManager(config=_build_state_config())
    assert manager.current_step == 0
    assert manager.current_episode == 1
    assert manager.is_initialized is False


def test_state_manager_load_data() -> None:
    """Load DataBatch and build an initial state."""

    manager = StateManager(config=_build_state_config())
    manager.load_data(_build_batch())
    manager.episode_length = 40
    manager.total_timesteps = 40

    state = manager.initialise_state()
    assert state
    assert manager.is_initialized is True
    assert len(manager.state_df) == 10


def test_state_manager_state_format() -> None:
    """Verify state output dictionary structure and array lengths."""

    manager = StateManager(config=_build_state_config())
    manager.load_data(_build_batch())
    manager.episode_length = 40
    manager.total_timesteps = 40

    state = manager.initialise_state()
    assert set(state.keys()) == {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "wap",
        "count",
    }
    assert all(len(values) == 10 for values in state.values())


def test_state_manager_stepping() -> None:
    """Advance through data and verify rolling-window progression."""

    manager = StateManager(config=_build_state_config())
    manager.load_data(_build_batch())
    manager.episode_length = 40
    manager.total_timesteps = 40
    manager.initialise_state()

    first_close_array = manager.state["close"].copy()
    manager.state_step(action=1)
    second_close_array = manager.state["close"]

    assert not np.array_equal(first_close_array, second_close_array)
    assert manager.current_step == 1


def test_state_manager_scaling() -> None:
    """Verify scaled numeric columns are normalized between 0 and 1."""

    manager = StateManager(config=_build_state_config())
    manager.load_data(_build_batch())
    manager.episode_length = 40
    manager.total_timesteps = 40
    state = manager.initialise_state()

    for column in ["open", "high", "low", "close", "volume", "wap", "count"]:
        values = state[column]
        assert np.all(values >= 0.0)
        assert np.all(values <= 1.0)


def test_state_manager_custom_variables() -> None:
    """Inject custom logic and verify custom variables are carried in state."""

    manager = StateManager(
        config=_build_state_config(), custom_logic=_CustomLogicStub()
    )
    manager.load_data(_build_batch())
    manager.episode_length = 40
    manager.total_timesteps = 40
    initial_state = manager.initialise_state()

    assert "current_position" in initial_state
    assert "trade_change" in initial_state

    manager.state_step(action=2)
    assert float(manager.state["current_position"][-1]) == 2.0
    assert float(manager.state["trade_change"][-1]) == 1.0


def test_state_manager_vs_state_builder() -> None:
    """Compare StateManager output with legacy StateBuilder exactly."""

    batch = _build_batch()
    dataframe = batch.to_dataframe().rename(columns={"timestamp": "date"})
    pipeline, config = _build_pipeline_and_config()

    state_builder = StateBuilder(config, pipeline, _CustomLogicStub())
    state_builder.final_dataframe = dataframe
    state_builder.episode_length = 40
    state_builder.total_timesteps = 40
    state_builder.initialise_state()

    manager = StateManager(
        config=_build_state_config(), custom_logic=_CustomLogicStub()
    )
    manager.load_dataframe(dataframe)
    manager.episode_length = 40
    manager.total_timesteps = 40
    manager.initialise_state()

    assert set(manager.state.keys()) == set(state_builder.state.keys())
    for key in state_builder.state:
        np.testing.assert_allclose(manager.state[key], state_builder.state[key])

    state_builder.state_step(1)
    manager.state_step(1)
    for key in state_builder.state:
        np.testing.assert_allclose(manager.state[key], state_builder.state[key])
