"""Unit tests for extracted StateManager and ScalerWrapper."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from algotrading.src.data_pipeline.state import ScalerWrapper, StateConfig, StateManager
from algotrading.src.data_pipeline.types import (
    DataBatch,
    DataFrequency,
    DataRecord,
    DataType,
)


@dataclass
class _CustomLogicStub:
    """Minimal custom logic for deterministic state variable updates."""

    def initialise_variables(self) -> dict[str, float]:
        return {"current_position": 0.0, "trade_change": 0.0}

    def step(
        self,
        action: int,
        state_df,
        custom_variable_dict: dict[str, np.ndarray],
        terminated: bool,
    ) -> dict[str, np.ndarray]:
        position = np.append(custom_variable_dict["current_position"], float(action))
        trade_change = np.append(
            custom_variable_dict["trade_change"],
            float(action != 0),
        )
        return {
            "current_position": position,
            "trade_change": trade_change,
        }


def _build_batch(record_count: int = 8) -> DataBatch:
    start = datetime(2024, 1, 1, 9, 30, 0)
    records: list[DataRecord] = []
    for index in range(record_count):
        ts = start + timedelta(seconds=5 * index)
        records.append(
            DataRecord(
                timestamp=ts,
                data_type=DataType.MARKET_BAR,
                symbol="AMD",
                payload={
                    "open": 100.0 + index,
                    "high": 101.0 + index,
                    "low": 99.0 + index,
                    "close": 100.5 + index,
                    "volume": 1000 + (10 * index),
                    "wap": 100.2 + index,
                    "count": 10 + index,
                },
                source_id="test_source",
                frequency=DataFrequency.SECOND_5,
            )
        )

    return DataBatch(
        records=records,
        start_time=records[0].timestamp,
        end_time=records[-1].timestamp,
        data_type=DataType.MARKET_BAR,
        symbol="AMD",
    )


def _build_state_config() -> StateConfig:
    return StateConfig(
        window_size=3,
        columns={
            "date": (True, False),
            "open": (False, True),
            "high": (False, True),
            "low": (False, True),
            "close": (False, True),
            "volume": (False, False),
            "wap": (False, True),
            "count": (False, False),
        },
        scaler_type="MinMaxScaler",
    )


def test_scaler_wrapper_supports_expected_scalers() -> None:
    """Ensure ScalerWrapper can construct all supported scaler types."""

    values = np.array([[1.0], [2.0], [3.0]])

    for scaler_name in ["MinMaxScaler", "StandardScaler", "RobustScaler"]:
        wrapper = ScalerWrapper(scaler_name)
        transformed = wrapper.fit_transform(values)
        assert transformed.shape == values.shape


def test_state_manager_load_data_and_initialise_state() -> None:
    """Verify DataBatch loading and initial state construction."""

    manager = StateManager(_build_state_config(), custom_logic=_CustomLogicStub())
    manager.load_data(_build_batch())
    manager.episode_length = 5
    manager.total_timesteps = 5

    state = manager.initialise_state()

    assert manager.is_initialized is True
    assert "open" in state
    assert "volume" in state
    assert "current_position" in state
    assert len(state["close"]) == 3


def test_state_manager_state_step_advances_and_updates_custom_variables() -> None:
    """Ensure stepping updates counters, state window, and custom variables."""

    manager = StateManager(_build_state_config(), custom_logic=_CustomLogicStub())
    manager.load_data(_build_batch())
    manager.episode_length = 5
    manager.total_timesteps = 5
    manager.initialise_state()

    state, terminated = manager.state_step(action=2)

    assert terminated is False
    assert manager.current_step == 1
    assert state["current_position"][-1] == 2.0
    assert state["trade_change"][-1] == 1.0


def test_state_manager_scales_configured_columns() -> None:
    """Verify scaled columns are normalized for the active window."""

    manager = StateManager(_build_state_config(), custom_logic=_CustomLogicStub())
    manager.load_data(_build_batch())
    manager.episode_length = 5
    manager.total_timesteps = 5

    state = manager.initialise_state()

    assert float(np.min(state["open"])) >= 0.0
    assert float(np.max(state["open"])) <= 1.0
    assert float(np.min(state["close"])) >= 0.0
    assert float(np.max(state["close"])) <= 1.0
