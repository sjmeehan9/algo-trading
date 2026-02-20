"""Tests for the consolidated BaseMetrics implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from algotrading.src.metrics.base_metrics import BaseMetrics
from algotrading.src.reward_functions.profit_seeker import ProfitSeeker
from algotrading.src.strategies.profit_metrics import ProfitMetrics


def _build_config(task_selection: str = "task2") -> dict[str, str]:
    """Build a minimal config dict for metrics tests."""

    return {"task_selection": task_selection}


def _build_reward_pipeline() -> dict[str, dict[str, dict[str, str]]]:
    """Build a pipeline dict with reward wrapper settings."""

    return {
        "pipeline": {
            "model": {
                "reward_wrapper_path": "",
                "reward_wrapper_filename": "missing_reward_wrapper_file_12345",
            }
        }
    }


def _build_strategy_pipeline() -> dict[str, dict[str, dict[str, str]]]:
    """Build a pipeline dict with strategy wrapper settings."""

    return {
        "pipeline": {
            "strategy": {
                "strategy_wrapper_path": "",
                "strategy_wrapper_filename": "missing_strategy_wrapper_file_12345",
            }
        }
    }


@pytest.mark.parametrize(
    "current_position,action,terminated,expected",
    [
        (0, 0, True, "hold_nothing"),
        (1, 0, True, "sell_position"),
        (2, 0, True, "buyback_short"),
        (0, 0, False, "hold_nothing"),
        (0, 1, False, "buy_long"),
        (0, 2, False, "sell_short"),
        (1, 0, False, "hold_long_position"),
        (1, 1, False, "false_buy"),
        (2, 2, False, "false_sell"),
    ],
)
def test_determine_action_type(
    current_position: int, action: int, terminated: bool, expected: str
) -> None:
    """Ensure action types are determined correctly."""

    metrics = BaseMetrics(_build_config(), {})
    metrics.current_position = current_position

    metrics.determine_action_type(action, terminated)

    assert metrics.action_type == expected


def test_state_step_updates_trade_and_profit() -> None:
    """Validate trade change and running profit updates."""

    metrics = BaseMetrics(_build_config(), {})
    state_df = pd.DataFrame({"close": [100.0, 102.0]})

    custom_variable_dict = {
        "current_position": np.array([0, 0]),
        "trade_change": np.array([0.0, 0.0]),
        "running_profit": np.array([0.0, 0.0]),
    }

    updated = metrics.state_step(1, state_df, custom_variable_dict, False)

    strike_price = 100.0
    strike_buy = strike_price / metrics.price_penalty
    new_sell = 102.0 * metrics.price_penalty
    expected_trade_change = (new_sell / strike_buy - 1) * 100

    assert updated["current_position"][-1] == 1
    assert np.isclose(updated["trade_change"][-1], expected_trade_change)
    assert np.isclose(updated["running_profit"][-1], expected_trade_change)


def test_reset_env_globals() -> None:
    """Ensure session globals are reset to defaults."""

    metrics = BaseMetrics(_build_config(), {})
    metrics.PRICE_PAID = 10.0
    metrics.SET_PROFIT = 5.0

    metrics.reset_env_globals()

    assert metrics.PRICE_PAID == 0.0
    assert metrics.SET_PROFIT == 0.0


def test_profit_seeker_reward_calculation() -> None:
    """Validate ProfitSeeker reward computation with a fixed state."""

    metrics = ProfitSeeker(_build_config(), _build_reward_pipeline())
    metrics.action_type = "hold_long_position"

    state = {
        "trade_change": np.array([0.0, 2.0]),
        "running_profit": np.array([0.0, 3.0]),
    }

    reward = metrics.calculate_reward(state)

    assert np.isclose(reward, 38.5)


def test_profit_metrics_predict_default_action() -> None:
    """Ensure ProfitMetrics returns the default action and empty states."""

    metrics = ProfitMetrics(_build_config(), _build_strategy_pipeline())

    action, states = metrics.predict({"close": np.array([1.0])}, {"foo": "bar"})

    assert action == 0
    assert states == {}
