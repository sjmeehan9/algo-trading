"""Mock RL trainer for deterministic offline training/backtest tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable

import pandas as pd
from algotrading.src.trainers import (
    EvaluationResult,
    RLTrainer,
    TrainingConfig,
    TrainingResult,
)
from gymnasium import Env


class MockRLTrainer(RLTrainer):
    """Simple in-memory trainer implementation for test orchestration."""

    def __init__(self) -> None:
        self._is_trained = False
        self._model_type = "mock_rl"
        self._predict_action = 0
        self._training_result = TrainingResult(
            timesteps_trained=100,
            episodes_completed=1,
            final_reward=0.0,
            training_time_seconds=0.01,
            model_path=None,
        )
        self.loaded_path: str | None = None
        self.saved_path: str | None = None
        self.train_calls = 0
        self.predict_calls = 0
        self.model_created = False

    def set_predict_action(self, action: int) -> None:
        """Set the action returned by ``predict``."""

        self._predict_action = action

    def set_training_result(self, result: TrainingResult) -> None:
        """Set the training result returned by ``train``."""

        self._training_result = result

    def create_model(self, env: Env, config: TrainingConfig) -> None:
        del env, config
        self.model_created = True

    def train(
        self,
        config: TrainingConfig,
        callback: Callable[[dict[str, object]], None] | None = None,
    ) -> TrainingResult:
        self.train_calls += 1
        self._is_trained = True

        if callback is not None:
            callback(
                {
                    "timesteps": config.total_timesteps,
                    "reward": self._training_result.final_reward,
                    "episodes": self._training_result.episodes_completed,
                }
            )

        return replace(
            self._training_result,
            timesteps_trained=config.total_timesteps,
        )

    def evaluate(self, env: Env, n_episodes: int = 10) -> EvaluationResult:
        del env
        return EvaluationResult(
            episodes=n_episodes,
            mean_reward=0.0,
            std_reward=0.0,
            mean_episode_length=1.0,
            results_df=pd.DataFrame(
                {"episode": list(range(n_episodes)), "reward": [0.0] * n_episodes}
            ),
        )

    def save(self, filepath: str) -> None:
        model_path = Path(filepath)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"mock-rl-model")
        self.saved_path = filepath

    def load(self, filepath: str, env: Env | None = None) -> None:
        del env
        self._is_trained = True
        self.loaded_path = filepath

    def predict(
        self,
        observation: dict[str, object],
        deterministic: bool = True,
    ) -> tuple[int, dict[str, object]]:
        del observation, deterministic
        self.predict_calls += 1
        self.ensure_trained()
        return self._predict_action, {"states": None}

    @property
    def model_type(self) -> str:
        return self._model_type

    @property
    def is_trained(self) -> bool:
        return self._is_trained
