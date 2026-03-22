"""Tests for the consolidated trading environments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from algotrading.src.envs.signal_integration import SignalConfig, SignalIntegration
from algotrading.src.envs.strategy_env import StrategyEnv
from algotrading.src.envs.trading_env import TradingEnv
from algotrading.src.models.signals import ModelSignal, SignalMetadata, SignalType
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
        self.state_df = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2026-01-01T14:30:00Z",
                        "2026-01-01T14:31:00Z",
                        "2026-01-01T14:32:00Z",
                    ],
                    utc=True,
                )
            }
        )

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
        next_ts = self.state_df["date"].iloc[-1] + pd.Timedelta(minutes=1)
        self.state_df = pd.DataFrame({"date": [next_ts]})

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


class DummyAlignmentService:
    """Simple alignment service test double."""

    def __init__(self, results: dict[str, object]) -> None:
        self._results = results

    def align(self, timestamp: datetime, model_ids: list[str]) -> object:
        _ = timestamp
        signals = {model_id: self._results.get(model_id) for model_id in model_ids}

        class _Aligned:
            def __init__(self, payload: dict[str, object]) -> None:
                self.signals = payload

        return _Aligned(signals)


class DummyAlignedSignal:
    """Minimal aligned signal object used by DummyAlignmentService."""

    def __init__(self, signal: ModelSignal | None, is_stale: bool) -> None:
        self.signal = signal
        self.is_stale = is_stale


def _build_signal(model_id: str, value: float, confidence: float = 0.8) -> ModelSignal:
    """Construct a deterministic test signal."""

    return ModelSignal(
        timestamp=datetime(2026, 1, 1, 14, 32, tzinfo=timezone.utc),
        signal_type=SignalType.SENTIMENT,
        value=value,
        confidence=confidence,
        symbol="AAPL",
        metadata=SignalMetadata(model_id=model_id, model_type="ml"),
    )


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


def test_env_with_signals_wraps_market_and_signals_in_observation() -> None:
    """Ensure signal-enabled env returns market/signal structured observations."""

    pipeline = _build_pipeline()
    custom_logic = DummyCustomLogic()
    state_builder = DummyStateBuilder(pipeline, custom_logic)

    aligned_signal = DummyAlignedSignal(_build_signal("sentiment_model", 0.6), False)
    alignment_service = DummyAlignmentService({"sentiment_model": aligned_signal})
    integration = SignalIntegration(
        alignment_service=alignment_service,  # type: ignore[arg-type]
        signal_configs=[
            SignalConfig(
                model_id="sentiment_model",
                signal_key="sentiment",
                default_value=0.0,
                include_confidence=True,
                include_staleness=True,
            )
        ],
    )

    env = TradingEnv(state_builder, signal_integration=integration)
    obs, info = env.reset()

    assert isinstance(env.observation_space, Dict)
    assert "market" in env.observation_space.spaces
    assert "signals" in env.observation_space.spaces
    assert "market" in obs
    assert "signals" in obs
    assert obs["signals"].shape == (3,)
    assert info == {}


def test_env_step_includes_signal_debug_info() -> None:
    """Ensure step info includes signal metadata when integration is enabled."""

    pipeline = _build_pipeline()
    custom_logic = DummyCustomLogic()
    state_builder = DummyStateBuilder(pipeline, custom_logic)

    aligned_signal = DummyAlignedSignal(
        _build_signal("sentiment_model", 0.4, 0.9), False
    )
    alignment_service = DummyAlignmentService({"sentiment_model": aligned_signal})
    integration = SignalIntegration(
        alignment_service=alignment_service,  # type: ignore[arg-type]
        signal_configs=[
            SignalConfig(
                model_id="sentiment_model",
                signal_key="sentiment",
                default_value=-0.5,
                include_confidence=True,
                include_staleness=True,
            )
        ],
    )

    env = TradingEnv(state_builder, signal_integration=integration)
    obs, _, _, _, info = env.step(1)

    assert "market" in obs
    assert "signals" in obs
    assert "signals" in info
    assert "sentiment" in info["signals"]
    assert info["signals"]["sentiment"]["value"] == 0.4
    assert info["signals"]["sentiment"]["is_stale"] is False
