"""Stable Baselines 3 implementation of the RLTrainer interface."""

from __future__ import annotations

import logging
import pickle
import time
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from algotrading.src.trainers.exceptions import ModelLoadError, TrainingError
from algotrading.src.trainers.rl_trainer import (
    EvaluationResult,
    RLTrainer,
    TrainingConfig,
    TrainingResult,
)
from gymnasium import Env
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback


class SB3Algorithm(str, Enum):
    """Supported Stable Baselines 3 algorithms."""

    PPO = "ppo"
    DQN = "dqn"
    A2C = "a2c"


class _ProgressCallback(BaseCallback):
    """Translate SB3 callback events into trainer progress dictionaries."""

    def __init__(
        self,
        callback: Callable[[dict[str, object]], None],
        trainer_name: str,
    ) -> None:
        super().__init__()
        self._callback = callback
        self._trainer_name = trainer_name
        self.last_reward: float = 0.0
        self.episodes_completed: int = 0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        if rewards is not None:
            reward_array = np.asarray(rewards).reshape(-1)
            if reward_array.size > 0:
                self.last_reward = float(reward_array[-1])

        dones = self.locals.get("dones")
        if dones is not None:
            done_array = np.asarray(dones, dtype=bool).reshape(-1)
            self.episodes_completed += int(done_array.sum())

        payload: dict[str, object] = {
            "trainer": self._trainer_name,
            "timesteps": int(self.num_timesteps),
            "reward": self.last_reward,
            "episodes": self.episodes_completed,
        }
        self._callback(payload)
        return True


class StableBaselines3Trainer(RLTrainer):
    """RLTrainer implementation backed by Stable Baselines 3.

    Supports PPO and DQN currently, with A2C included in the algorithm enum for
    forward compatibility.
    """

    _MODEL_CLASS_MAP: dict[SB3Algorithm, type[BaseAlgorithm]] = {
        SB3Algorithm.PPO: PPO,
        SB3Algorithm.DQN: DQN,
        SB3Algorithm.A2C: A2C,
    }

    def __init__(
        self,
        algorithm: SB3Algorithm,
        policy: str = "MlpPolicy",
    ) -> None:
        """Initialize a Stable Baselines 3 trainer.

        Args:
            algorithm: SB3 algorithm identifier.
            policy: SB3 policy class name.
        """

        self.algorithm = algorithm
        self.policy = policy
        self._model: BaseAlgorithm | None = None
        self._is_trained = False
        self._logger = logging.getLogger(__name__)

    def _get_model_class(self) -> type[BaseAlgorithm]:
        """Return the SB3 model class matching the configured algorithm."""

        try:
            return self._MODEL_CLASS_MAP[self.algorithm]
        except KeyError as exc:
            raise TrainingError(
                message=f"Unsupported SB3 algorithm: {self.algorithm}",
                trainer_name=self.model_type,
                stage="create_model",
                reason="algorithm_not_supported",
            ) from exc

    def _replay_buffer_path(self, filepath: str) -> Path:
        """Return sidecar replay-buffer filepath for DQN models."""

        return Path(f"{filepath}_replay_buffer.pkl")

    def _resolve_model_path(self, filepath: str) -> str:
        """Resolve model path with optional `.zip` suffix fallback."""

        candidate = Path(filepath)
        if candidate.exists():
            return str(candidate)

        zip_candidate = Path(f"{filepath}.zip")
        if zip_candidate.exists():
            return str(zip_candidate)

        return str(candidate)

    def _wrap_callback(
        self,
        callback: Callable[[dict[str, object]], None],
    ) -> _ProgressCallback:
        """Wrap a generic callback in an SB3-compatible callback class."""

        return _ProgressCallback(callback=callback, trainer_name=self.model_type)

    def create_model(self, env: Env, config: TrainingConfig) -> None:
        """Create an SB3 model using trainer algorithm and config settings."""

        self.validate_env(env)
        model_class = self._get_model_class()

        model_kwargs: dict[str, object] = {
            "policy": self.policy,
            "env": env,
            "tensorboard_log": config.tensorboard_log,
            "learning_rate": config.learning_rate or 0.0003,
            "batch_size": config.batch_size or 64,
        }

        if self.algorithm == SB3Algorithm.PPO and config.n_steps is not None:
            model_kwargs["n_steps"] = config.n_steps

        if config.custom_params:
            model_kwargs.update(config.custom_params)

        try:
            self._model = model_class(**model_kwargs)
            self._is_trained = False
        except Exception as exc:  # pragma: no cover - defensive error wrapping
            raise TrainingError(
                message="Failed to create SB3 model.",
                trainer_name=self.model_type,
                stage="create_model",
                reason=str(exc),
            ) from exc

    def train(
        self,
        config: TrainingConfig,
        callback: Callable[[dict[str, object]], None] | None = None,
        *,
        reset_num_timesteps: bool = True,
    ) -> TrainingResult:
        """Train the configured SB3 model and return summary metadata.

        Args:
            config: Training configuration including the timestep budget.
            callback: Optional progress callback.
            reset_num_timesteps: When ``False``, continue the model's existing
                timestep counter instead of resetting it. Used to warm-start
                (continue) training from a previously trained artifact.
        """

        if self._model is None:
            raise TrainingError(
                message="Model must be created before training.",
                trainer_name=self.model_type,
                stage="train",
                reason="model_not_created",
            )

        start_time = time.time()
        progress_callback: _ProgressCallback | None = (
            self._wrap_callback(callback) if callback else None
        )

        try:
            self._model.learn(
                total_timesteps=config.total_timesteps,
                callback=progress_callback,
                progress_bar=False,
                reset_num_timesteps=reset_num_timesteps,
            )
        except Exception as exc:  # pragma: no cover - defensive error wrapping
            raise TrainingError(
                message="SB3 model training failed.",
                trainer_name=self.model_type,
                stage="learn",
                reason=str(exc),
            ) from exc

        self._is_trained = True
        training_time = time.time() - start_time

        episodes = progress_callback.episodes_completed if progress_callback else 0
        final_reward = progress_callback.last_reward if progress_callback else 0.0

        return TrainingResult(
            timesteps_trained=config.total_timesteps,
            episodes_completed=episodes,
            final_reward=final_reward,
            training_time_seconds=training_time,
            model_path=None,
        )

    def evaluate(self, env: Env, n_episodes: int = 10) -> EvaluationResult:
        """Evaluate the trained model across episodes and aggregate metrics."""

        self.ensure_trained()
        self.validate_env(env)

        if self._model is None:
            raise TrainingError(
                message="Model is not available for evaluation.",
                trainer_name=self.model_type,
                stage="evaluate",
                reason="model_not_loaded",
            )

        episode_rewards: list[float] = []
        episode_lengths: list[int] = []
        episode_rows: list[dict[str, float | int]] = []

        for episode in range(n_episodes):
            observation, _ = env.reset()
            done = False
            total_reward = 0.0
            steps = 0

            while not done:
                action, _states = self._model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                steps += 1
                done = bool(terminated or truncated)

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            episode_rows.append(
                {
                    "episode": episode + 1,
                    "reward": total_reward,
                    "episode_length": steps,
                }
            )

        rewards_array = np.asarray(episode_rewards, dtype=np.float64)
        lengths_array = np.asarray(episode_lengths, dtype=np.float64)

        return EvaluationResult(
            episodes=n_episodes,
            mean_reward=float(rewards_array.mean()) if rewards_array.size else 0.0,
            std_reward=float(rewards_array.std()) if rewards_array.size else 0.0,
            mean_episode_length=(
                float(lengths_array.mean()) if lengths_array.size else 0.0
            ),
            results_df=pd.DataFrame(episode_rows),
        )

    def save(self, filepath: str) -> None:
        """Persist model and optional DQN replay buffer artifacts."""

        self.ensure_trained()

        if self._model is None:
            raise TrainingError(
                message="Model is not available for saving.",
                trainer_name=self.model_type,
                stage="save",
                reason="model_not_loaded",
            )

        self._model.save(filepath)

        if self.algorithm == SB3Algorithm.DQN and hasattr(self._model, "replay_buffer"):
            replay_buffer = getattr(self._model, "replay_buffer")
            replay_path = self._replay_buffer_path(filepath)
            replay_path.parent.mkdir(parents=True, exist_ok=True)
            with replay_path.open("wb") as file_handle:
                pickle.dump(replay_buffer, file_handle)

    def load(self, filepath: str, env: Env | None = None) -> None:
        """Load model artifact and optional DQN replay buffer."""

        model_class = self._get_model_class()
        model_path = self._resolve_model_path(filepath)

        if env is not None:
            self.validate_env(env)

        try:
            self._model = model_class.load(model_path, env=env)
        except Exception as exc:
            raise ModelLoadError(
                message="Failed to load SB3 model.",
                trainer_name=self.model_type,
                filepath=model_path,
                reason=str(exc),
            ) from exc

        if self.algorithm == SB3Algorithm.DQN:
            replay_path = self._replay_buffer_path(filepath)
            if replay_path.exists() and self._model is not None:
                with replay_path.open("rb") as file_handle:
                    replay_buffer = pickle.load(file_handle)
                setattr(self._model, "replay_buffer", replay_buffer)
                self._logger.info("Loaded DQN replay buffer from %s", replay_path)

        self._is_trained = True

    def predict(
        self,
        observation: dict[str, object],
        deterministic: bool = True,
    ) -> tuple[int, dict[str, object]]:
        """Predict an action for a given observation."""

        self.ensure_trained()

        if self._model is None:
            raise TrainingError(
                message="Model is not available for prediction.",
                trainer_name=self.model_type,
                stage="predict",
                reason="model_not_loaded",
            )

        action, states = self._model.predict(observation, deterministic=deterministic)
        action_value = int(np.asarray(action).reshape(-1)[0])
        return action_value, {"states": states}

    @property
    def model_type(self) -> str:
        """Return the configured trainer model type."""

        return self.algorithm.value

    @property
    def is_trained(self) -> bool:
        """Return whether the trainer currently has a trained/loaded model."""

        return self._is_trained


__all__ = ["SB3Algorithm", "StableBaselines3Trainer"]
