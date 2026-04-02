"""Integration tests for signal-enhanced trading environments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import pytest
from algotrading.src.envs.signal_integration import SignalConfig, SignalIntegration
from algotrading.src.envs.trading_env import TradingEnv
from algotrading.src.models.signals import ModelSignal, SignalMetadata, SignalType
from algotrading.src.trainers import (
    SB3Algorithm,
    StableBaselines3Trainer,
    TrainingConfig,
)
from gymnasium.wrappers import FlattenObservation


@dataclass
class _CustomLogic:
    """Minimal custom logic for environment integration tests."""

    CUSTOM_VARIABLES = {"current_position": (0, 1, np.int64)}

    def calculate_reward(self, state: dict[str, Any]) -> float:
        """Return deterministic reward."""

        _ = state
        return 1.0

    def reset_env_globals(self) -> None:
        """No-op reset used in tests."""


class _StateBuilder:
    """StateBuilder test double with rolling timestamps."""

    def __init__(self, pipeline: dict[str, Any]) -> None:
        self.pipeline = pipeline
        self.custom_logic = _CustomLogic()
        self.state_counters = {"step": 0, "window": 0, "episode": 1}
        self.terminated = False
        self.timed_out = False
        self.state = self._build_state()
        self.state_df = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2026-01-02T14:30:00Z",
                        "2026-01-02T14:31:00Z",
                        "2026-01-02T14:32:00Z",
                    ],
                    utc=True,
                )
            }
        )

    def _build_state(self) -> dict[str, np.ndarray]:
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
        _ = action
        self.state_counters["step"] += 1
        next_ts = self.state_df["date"].iloc[-1] + pd.Timedelta(minutes=1)
        self.state_df = pd.DataFrame({"date": [next_ts]})

    def update_episode_counter(self) -> None:
        self.state_counters["episode"] += 1

    def initialise_state(self) -> None:
        self.state = self._build_state()


class _AlignedSignal:
    """Aligned signal payload used by alignment test double."""

    def __init__(self, signal: ModelSignal | None, is_stale: bool) -> None:
        self.signal = signal
        self.is_stale = is_stale


class _AlignmentService:
    """Minimal alignment service that always returns one aligned signal."""

    def align(self, timestamp: datetime, model_ids: list[str]) -> object:
        signal = ModelSignal(
            timestamp=(
                timestamp
                if timestamp.tzinfo
                else timestamp.replace(tzinfo=timezone.utc)
            ),
            signal_type=SignalType.SENTIMENT,
            value=0.25,
            confidence=0.85,
            symbol="AAPL",
            metadata=SignalMetadata(model_id="sentiment_model", model_type="ml"),
        )

        class _Aligned:
            def __init__(self, payload: dict[str, _AlignedSignal]) -> None:
                self.signals = payload

        return _Aligned({model_ids[0]: _AlignedSignal(signal=signal, is_stale=False)})


def _pipeline() -> dict[str, Any]:
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


def test_signal_enhanced_environment_runs_multiple_steps() -> None:
    """Ensure enhanced environment can reset and step repeatedly with signal features."""

    state_builder = _StateBuilder(_pipeline())
    integration = SignalIntegration(
        alignment_service=_AlignmentService(),  # type: ignore[arg-type]
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

    assert "market" in obs
    assert "signals" in obs
    assert obs["signals"].shape == (3,)
    assert info == {}

    for _ in range(3):
        obs, reward, terminated, truncated, step_info = env.step(1)
        assert "market" in obs
        assert "signals" in obs
        assert obs["signals"].shape == (3,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "signals" in step_info
        assert "sentiment" in step_info["signals"]


@pytest.mark.slow
def test_signal_enhanced_environment_supports_short_training_loop() -> None:
    """Signal-enhanced TradingEnv should support short PPO training and inference."""

    state_builder = _StateBuilder(_pipeline())
    integration = SignalIntegration(
        alignment_service=_AlignmentService(),  # type: ignore[arg-type]
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
    signal_env = TradingEnv(state_builder, signal_integration=integration)
    env = FlattenObservation(signal_env)

    trainer = StableBaselines3Trainer(
        algorithm=SB3Algorithm.PPO,
        policy="MlpPolicy",
    )
    config = TrainingConfig(
        total_timesteps=96,
        learning_rate=0.0003,
        batch_size=32,
        n_steps=32,
        custom_params={"verbose": 0},
    )

    trainer.create_model(env, config)
    result = trainer.train(config)

    observation, _ = env.reset()
    action, info = trainer.predict(observation)

    assert result.timesteps_trained == 96
    assert trainer.is_trained is True
    assert action in (0, 1, 2)
    assert "states" in info
