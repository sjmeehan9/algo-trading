"""Integration parity tests for StateManager extraction from StateBuilder."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from algotrading.src.data_pipeline.state import StateConfig, StateManager
from algotrading.src.data_sourcing.state_builder import StateBuilder


@dataclass
class _CustomLogicStub:
    """Custom logic stub mirroring StateBuilder custom variable contract."""

    def initialise_variables(self) -> dict[str, float]:
        return {"current_position": 0.0, "trade_change": 0.0}

    def step(
        self,
        action: int,
        state_df: pd.DataFrame,
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


def _build_dataframe(row_count: int = 40) -> pd.DataFrame:
    start = datetime(2024, 1, 2, 9, 30)
    rows = []
    for index in range(row_count):
        rows.append(
            {
                "date": (start + timedelta(seconds=5 * index)).strftime(
                    "%Y%m%d %H:%M:%S US/Eastern"
                ),
                "open": 100.0 + index,
                "high": 101.0 + index,
                "low": 99.0 + index,
                "close": 100.5 + index,
                "volume": 1000 + (10 * index),
                "wap": 100.2 + index,
                "count": 10 + index,
            }
        )
    return pd.DataFrame(rows)


def _build_pipeline() -> dict:
    return {
        "pipeline": {
            "filename": "pipeline_0001",
            "contract_info": {
                "symbol": "AMD",
                "primaryExchange": "NASDAQ",
            },
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


def _build_config() -> dict:
    return {
        "task_selection": "task2",
        "data_mode": "historical",
        "data_path": "",
        "training_date_list": [],
        "backtest_date_list": [],
    }


def test_state_manager_matches_state_builder_output_for_initial_and_step() -> None:
    """StateManager output should match StateBuilder state format and values."""

    dataframe = _build_dataframe()
    pipeline = _build_pipeline()
    config = _build_config()

    state_builder = StateBuilder(config, pipeline, _CustomLogicStub())
    state_builder.final_dataframe = dataframe
    state_builder.episode_length = (
        len(dataframe) - pipeline["pipeline"]["state_data_config"]["past_events"]
    )
    state_builder.total_timesteps = state_builder.episode_length
    state_builder.initialise_state()

    manager = StateManager(
        StateConfig(
            window_size=pipeline["pipeline"]["state_data_config"]["past_events"],
            columns={
                key: (bool(value[0]), bool(value[1]))
                for key, value in pipeline["pipeline"]["state_data_config"][
                    "columns"
                ].items()
            },
            scaler_type=pipeline["pipeline"]["state_data_config"]["scaler"],
        ),
        custom_logic=_CustomLogicStub(),
    )
    manager.load_dataframe(dataframe)
    manager.episode_length = state_builder.episode_length
    manager.total_timesteps = state_builder.total_timesteps
    manager.initialise_state()

    assert set(manager.state.keys()) == set(state_builder.state.keys())
    for key in state_builder.state:
        np.testing.assert_allclose(manager.state[key], state_builder.state[key])

    state_builder.state_step(2)
    manager.state_step(2)

    assert set(manager.state.keys()) == set(state_builder.state.keys())
    for key in state_builder.state:
        np.testing.assert_allclose(manager.state[key], state_builder.state[key])
