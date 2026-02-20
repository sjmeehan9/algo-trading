"""Tests for the consolidated trading environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from algotrading.src.envs.strategy_env import StrategyEnv
from algotrading.src.envs.trading_env import TradingEnv
from gymnasium.spaces import Dict


@dataclass
class DummyCustomLogic:
    """Custom logic stub for environment tests."""

    reward_value: float = 7.5
    reset_called: bool = False

    CUSTOM_VARIABLES = {"current_position": (0, 1, np.int64)}

    def calculate_reward(self, state: dict[str, Any]) -> float:
        """Return a deterministic reward for tests."""

        return self.reward_value

    def reset_env_globals(self) -> None:
        """Track reset calls."""

        self.reset_called = True


class DummyStateBuilder:
    """StateBuilder stub for testing environments."""

    def __init__(
        self, pipeline: dict[str, Any], custom_logic: DummyCustomLogic
    ) -> None:
        self.pipeline = pipeline
        self.custom_logic = custom_logic
        self.state_counters = {"step": 0, "window": 0, "episode": 1}
        self.terminated = False
        self.timed_out = False
        self.update_called = False
        self.initialise_called = False
        self.last_action: int | None = None
        self.state = self._build_state()

    def _build_state(self) -> dict[str, np.ndarray]:
        """Create a deterministic state dictionary for tests."""

        return {
            "open": np.array([100.0, 101.0, 102.0]),
            "high": np.array([101.0, 102.0, 103.0]),
            "low": np.array([99.0, 100.0, 101.0]),
            "close": np.array([100.5, 101.5, 102.5]),
            "volume": np.array([1000.0, 1100.0, 1050.0]),
            "count": np.array([10.0, 12.0, 11.0]),
            "current_position": np.array([0, 1, 0]),
        }

    def state_step(self, action: int) -> None:
        """Record step actions for assertions."""

        self.last_action = action
        self.state_counters["step"] += 1

    def update_episode_counter(self) -> None:
        """Track episode counter updates."""

        self.update_called = True

    def initialise_state(self) -> None:
        """Track state initialization calls."""

        self.initialise_called = True


def _build_pipeline() -> dict[str, Any]:
    """Create a minimal pipeline dict for environment tests."""

    return {
        "pipeline": {
            "state_data_config": {
                "columns": {
                    "date": [True, False],
                    "open": [False, True],
                    "high": [False, True],
                    "low": [False, True],
                    "close": [False, True],
                    "volume": [False, True],
                    "count": [False, True],
                }
            },
            "trading_config": {"stop_take": {"enabled": False}},
        }
    }


def test_obs_space_includes_expected_keys() -> None:
    """Ensure observation space includes state columns and custom variables."""

    pipeline = _build_pipeline()
    custom_logic = DummyCustomLogic()
    state_builder = DummyStateBuilder(pipeline, custom_logic)

    env = TradingEnv(state_builder)

    assert isinstance(env.observation_space, Dict)
    assert "open" in env.observation_space.spaces
    assert "close" in env.observation_space.spaces
    assert "current_position" in env.observation_space.spaces
    assert "date" not in env.observation_space.spaces


def test_trading_env_uses_reward_calculation() -> None:
    """Verify TradingEnv returns the custom reward value."""

    pipeline = _build_pipeline()
    custom_logic = DummyCustomLogic(reward_value=12.25)
    state_builder = DummyStateBuilder(pipeline, custom_logic)

    env = TradingEnv(state_builder)
    _, reward, _, _, _ = env.step(1)

    assert reward == 12.25
    assert state_builder.last_action == 1


def test_strategy_env_returns_dummy_reward() -> None:
    """Verify StrategyEnv returns the dummy reward value."""

    pipeline = _build_pipeline()
    custom_logic = DummyCustomLogic(reward_value=4.0)
    state_builder = DummyStateBuilder(pipeline, custom_logic)

    env = StrategyEnv(state_builder)
    _, reward, _, _, _ = env.step(2)

    assert reward == 0.0


def test_reset_updates_episode_and_custom_globals() -> None:
    """Ensure reset updates episode counters and resets custom globals."""

    pipeline = _build_pipeline()
    custom_logic = DummyCustomLogic()
    state_builder = DummyStateBuilder(pipeline, custom_logic)
    state_builder.terminated = True

    env = TradingEnv(state_builder)
    env.reset()

    assert state_builder.update_called is True
    assert custom_logic.reset_called is True
    assert state_builder.initialise_called is True
