"""Unit tests for RL trainer interface abstractions and exceptions."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
from algotrading.src.trainers import (
    EvaluationResult,
    InvalidEnvironmentError,
    ModelLoadError,
    ModelNotTrainedError,
    RLTrainer,
    TrainerError,
    TrainingConfig,
    TrainingError,
    TrainingResult,
)
from gymnasium import Env, spaces


class _BaseSuperHarness(RLTrainer):
    """Concrete harness delegating each abstract contract to super."""

    def create_model(self, env: Env, config: TrainingConfig) -> None:
        super().create_model(env, config)

    def train(self, config: TrainingConfig, callback=None) -> TrainingResult:
        return super().train(config, callback)

    def evaluate(self, env: Env, n_episodes: int = 10) -> EvaluationResult:
        return super().evaluate(env, n_episodes)

    def save(self, filepath: str) -> None:
        super().save(filepath)

    def load(self, filepath: str, env: Env | None = None) -> None:
        super().load(filepath, env)

    def predict(
        self,
        observation: dict[str, object],
        deterministic: bool = True,
    ) -> tuple[int, dict[str, object]]:
        return super().predict(observation, deterministic)

    @property
    def model_type(self) -> str:
        return super().model_type

    @property
    def is_trained(self) -> bool:
        return super().is_trained


class _DummyEnv(Env):
    """Minimal gymnasium environment for interface validation tests."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        self.observation_space = spaces.Dict(
            {"price": spaces.Box(low=0.0, high=10_000.0, shape=(1,), dtype=np.float32)}
        )
        self.action_space = spaces.Discrete(3)

    def reset(self, *, seed=None, options=None):  # type: ignore[override]
        super().reset(seed=seed)
        obs = {"price": np.array([1.0], dtype=np.float32)}
        info: dict[str, object] = {}
        return obs, info

    def step(self, action: int):  # type: ignore[override]
        obs = {"price": np.array([1.1], dtype=np.float32)}
        reward = 0.1
        terminated = True
        truncated = False
        info: dict[str, object] = {"action": action}
        return obs, reward, terminated, truncated, info


class _ConcreteTrainer(RLTrainer):
    """Simple concrete trainer used to exercise helper methods."""

    def __init__(self, *, trained: bool) -> None:
        self._trained = trained

    def create_model(self, env: Env, config: TrainingConfig) -> None:
        self.validate_env(env)

    def train(self, config: TrainingConfig, callback=None) -> TrainingResult:
        self._trained = True
        if callback:
            callback({"timesteps": config.total_timesteps, "episode": 1, "reward": 0.0})
        return TrainingResult(
            timesteps_trained=config.total_timesteps,
            episodes_completed=1,
            final_reward=0.0,
            training_time_seconds=0.01,
            model_path=None,
        )

    def evaluate(self, env: Env, n_episodes: int = 10) -> EvaluationResult:
        self.validate_env(env)
        return EvaluationResult(
            episodes=n_episodes,
            mean_reward=0.0,
            std_reward=0.0,
            mean_episode_length=1.0,
            results_df=None,
        )

    def save(self, filepath: str) -> None:
        _ = filepath

    def load(self, filepath: str, env: Env | None = None) -> None:
        _ = filepath
        if env is not None:
            self.validate_env(env)
        self._trained = True

    def predict(
        self,
        observation: dict[str, object],
        deterministic: bool = True,
    ) -> tuple[int, dict[str, object]]:
        self.ensure_trained()
        _ = deterministic
        return 1, {"obs_keys": list(observation.keys())}

    @property
    def model_type(self) -> str:
        return "mock"

    @property
    def is_trained(self) -> bool:
        return self._trained


def test_rl_trainer_abc_cannot_be_instantiated_directly() -> None:
    """RLTrainer is abstract and cannot be instantiated directly."""

    with pytest.raises(TypeError):
        RLTrainer()


def test_abstract_methods_raise_not_implemented_when_delegating_to_super() -> None:
    """Abstract members raise NotImplementedError via super delegation."""

    trainer = _BaseSuperHarness()
    env = _DummyEnv()
    config = TrainingConfig(total_timesteps=1)

    with pytest.raises(NotImplementedError):
        trainer.create_model(env, config)
    with pytest.raises(NotImplementedError):
        trainer.train(config)
    with pytest.raises(NotImplementedError):
        trainer.evaluate(env)
    with pytest.raises(NotImplementedError):
        trainer.save("model.zip")
    with pytest.raises(NotImplementedError):
        trainer.load("model.zip")
    with pytest.raises(NotImplementedError):
        trainer.predict({"price": np.array([1.0], dtype=np.float32)})
    with pytest.raises(NotImplementedError):
        _ = trainer.model_type
    with pytest.raises(NotImplementedError):
        _ = trainer.is_trained


def test_training_dataclasses_store_expected_values() -> None:
    """Training interface dataclasses store and expose expected fields."""

    training_config = TrainingConfig(
        total_timesteps=1_000,
        learning_rate=0.0003,
        batch_size=64,
        n_steps=128,
        tensorboard_log="/tmp/tb",
        custom_params={"gamma": 0.99},
    )
    training_result = TrainingResult(
        timesteps_trained=1_000,
        episodes_completed=10,
        final_reward=1.2,
        training_time_seconds=12.3,
        model_path="/tmp/model.zip",
    )
    eval_df = pd.DataFrame({"episode": [1, 2], "reward": [1.0, 1.4]})
    evaluation_result = EvaluationResult(
        episodes=2,
        mean_reward=1.2,
        std_reward=0.2,
        mean_episode_length=100.0,
        results_df=eval_df,
    )

    assert asdict(training_config)["total_timesteps"] == 1_000
    assert training_config.custom_params == {"gamma": 0.99}
    assert asdict(training_result)["final_reward"] == 1.2
    assert evaluation_result.results_df is eval_df
    assert evaluation_result.mean_reward == pytest.approx(1.2)


def test_exception_classes_preserve_context() -> None:
    """Trainer exception hierarchy stores context and readable formatting."""

    base = TrainerError(message="trainer failure", trainer_name="sb3")
    not_trained = ModelNotTrainedError(message="model missing", trainer_name="sb3")
    load_error = ModelLoadError(
        message="load failed",
        trainer_name="sb3",
        filepath="/tmp/model.zip",
        reason="file not found",
    )
    training_error = TrainingError(
        message="training failed",
        trainer_name="sb3",
        stage="learn",
        reason="nan loss",
    )
    invalid_env = InvalidEnvironmentError(
        message="invalid env",
        trainer_name="sb3",
        env_type="object",
        reason="not gym env",
    )

    assert str(base) == "[sb3] trainer failure"
    assert "model missing" in str(not_trained)
    assert "filepath=/tmp/model.zip" in str(load_error)
    assert "stage=learn" in str(training_error)
    assert "env_type=object" in str(invalid_env)


def test_ensure_trained_raises_when_model_not_ready() -> None:
    """ensure_trained raises ModelNotTrainedError if trainer is not ready."""

    trainer = _ConcreteTrainer(trained=False)

    with pytest.raises(ModelNotTrainedError):
        trainer.ensure_trained()


def test_validate_env_accepts_gymnasium_env() -> None:
    """validate_env accepts a compatible gymnasium environment."""

    trainer = _ConcreteTrainer(trained=True)
    trainer.validate_env(_DummyEnv())


def test_validate_env_rejects_non_env_object() -> None:
    """validate_env rejects objects that are not gymnasium environments."""

    trainer = _ConcreteTrainer(trained=True)

    with pytest.raises(InvalidEnvironmentError):
        trainer.validate_env(object())


def test_predict_returns_action_when_trained() -> None:
    """Concrete trainer predict returns action/info when model is trained."""

    trainer = _ConcreteTrainer(trained=True)
    action, info = trainer.predict({"price": np.array([1.0], dtype=np.float32)})

    assert action == 1
    assert info["obs_keys"] == ["price"]
