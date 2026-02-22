"""Abstract interface for reinforcement-learning model trainers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import pandas as pd
from algotrading.src.trainers.exceptions import (
    InvalidEnvironmentError,
    ModelNotTrainedError,
)
from gymnasium import Env


@dataclass(slots=True)
class TrainingConfig:
    """Configuration for RL training runs.

    Args:
        total_timesteps: Total timesteps to train.
        learning_rate: Optional optimizer learning rate.
        batch_size: Optional batch size.
        n_steps: Optional rollout steps per update.
        tensorboard_log: Optional tensorboard output directory.
        custom_params: Optional algorithm-specific parameters.
    """

    total_timesteps: int
    learning_rate: float | None = None
    batch_size: int | None = None
    n_steps: int | None = None
    tensorboard_log: str | None = None
    custom_params: dict[str, object] | None = None


@dataclass(slots=True)
class TrainingResult:
    """Outcome metadata from a completed training run.

    Args:
        timesteps_trained: Number of timesteps trained.
        episodes_completed: Number of episodes completed.
        final_reward: Final observed reward value.
        training_time_seconds: Total training duration in seconds.
        model_path: Optional saved model path.
    """

    timesteps_trained: int
    episodes_completed: int
    final_reward: float
    training_time_seconds: float
    model_path: str | None = None


@dataclass(slots=True)
class EvaluationResult:
    """Evaluation metrics for a trained RL model.

    Args:
        episodes: Number of evaluated episodes.
        mean_reward: Mean reward across episodes.
        std_reward: Standard deviation of rewards.
        mean_episode_length: Mean episode length.
        results_df: Optional detailed per-step or per-episode results.
    """

    episodes: int
    mean_reward: float
    std_reward: float
    mean_episode_length: float
    results_df: pd.DataFrame | None = None


class RLTrainer(ABC):
    """Abstract lifecycle interface for RL trainer implementations.

    Implementations are responsible for creating, training, evaluating,
    persisting, and running inference with a reinforcement-learning model.
    """

    @abstractmethod
    def create_model(self, env: Env, config: TrainingConfig) -> None:
        """Initialize a model instance bound to an environment.

        Args:
            env: Gymnasium environment used for training.
            config: Training configuration and hyperparameters.
        """

        raise NotImplementedError

    @abstractmethod
    def train(
        self,
        config: TrainingConfig,
        callback: Callable[[dict[str, object]], None] | None = None,
    ) -> TrainingResult:
        """Train the model for the configured timesteps.

        Args:
            config: Training configuration.
            callback: Optional callback receiving progress dictionaries.

        Returns:
            Aggregated training result metadata.
        """

        raise NotImplementedError

    @abstractmethod
    def evaluate(self, env: Env, n_episodes: int = 10) -> EvaluationResult:
        """Evaluate a trained model for a fixed number of episodes.

        Args:
            env: Gymnasium environment used for evaluation.
            n_episodes: Number of episodes to run.

        Returns:
            Aggregated evaluation metrics.
        """

        raise NotImplementedError

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Persist model artifacts to disk.

        Args:
            filepath: Destination model filepath.
        """

        raise NotImplementedError

    @abstractmethod
    def load(self, filepath: str, env: Env | None = None) -> None:
        """Load model artifacts from disk.

        Args:
            filepath: Source model filepath.
            env: Optional environment to bind to the loaded model.
        """

        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        observation: dict[str, object],
        deterministic: bool = True,
    ) -> tuple[int, dict[str, object]]:
        """Infer an action from an observation.

        Args:
            observation: Environment observation dictionary.
            deterministic: Whether to enforce deterministic action selection.

        Returns:
            A tuple of `(action, info)`.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return the model type identifier (for example `ppo` or `dqn`)."""

        raise NotImplementedError

    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """Return whether the model is ready for inference."""

        raise NotImplementedError

    def ensure_trained(self) -> None:
        """Ensure that model inference/training-dependent actions are valid.

        Raises:
            ModelNotTrainedError: If the model has not been trained or loaded.
        """

        if not self.is_trained:
            raise ModelNotTrainedError(
                message="Model has not been trained or loaded.",
                trainer_name=self.model_type,
            )

    def validate_env(self, env: Env) -> None:
        """Validate compatibility of a provided gymnasium environment.

        Args:
            env: Environment instance to validate.

        Raises:
            InvalidEnvironmentError: If the environment is incompatible.
        """

        if not isinstance(env, Env):
            raise InvalidEnvironmentError(
                message="Environment must be an instance of gymnasium.Env.",
                trainer_name=self.model_type,
                env_type=type(env).__name__,
            )

        if not hasattr(env, "action_space"):
            raise InvalidEnvironmentError(
                message="Environment is missing required action_space.",
                trainer_name=self.model_type,
                env_type=type(env).__name__,
            )

        if not hasattr(env, "observation_space"):
            raise InvalidEnvironmentError(
                message="Environment is missing required observation_space.",
                trainer_name=self.model_type,
                env_type=type(env).__name__,
            )


__all__ = [
    "RLTrainer",
    "TrainingConfig",
    "TrainingResult",
    "EvaluationResult",
]
