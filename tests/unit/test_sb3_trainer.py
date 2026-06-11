"""Unit tests for StableBaselines3Trainer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from algotrading.src.trainers import (
    ModelNotTrainedError,
    SB3Algorithm,
    StableBaselines3Trainer,
    TrainingConfig,
)
from gymnasium import Env, spaces


class _DummyEnv(Env):
    """Minimal environment for trainer tests."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        self.observation_space = spaces.Dict(
            {
                "price": spaces.Box(
                    low=0.0,
                    high=10_000.0,
                    shape=(1,),
                    dtype=np.float32,
                )
            }
        )
        self.action_space = spaces.Discrete(3)
        self._step_count = 0

    def reset(self, *, seed=None, options=None):  # type: ignore[override]
        super().reset(seed=seed)
        self._step_count = 0
        obs = {"price": np.array([1.0], dtype=np.float32)}
        info: dict[str, object] = {}
        return obs, info

    def step(self, action: int):  # type: ignore[override]
        del action
        self._step_count += 1
        obs = {"price": np.array([1.0 + self._step_count], dtype=np.float32)}
        reward = 1.0
        terminated = self._step_count >= 3
        truncated = False
        info: dict[str, object] = {}
        return obs, reward, terminated, truncated, info


@dataclass
class _FakeReplayBuffer:
    """Lightweight replay buffer stand-in for serialization tests."""

    marker: str


class _FakeModel:
    """SB3-like fake model used to isolate trainer tests."""

    last_load_path: str | None = None
    last_load_env: Env | None = None

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.saved_path: str | None = None
        self.replay_buffer = _FakeReplayBuffer(marker="initial")
        self.last_reset_num_timesteps: bool | None = None

    @classmethod
    def load(cls, path: str, env: Env | None = None):
        cls.last_load_path = path
        cls.last_load_env = env
        model = cls(policy="MlpPolicy", env=env)
        model.replay_buffer = _FakeReplayBuffer(marker="loaded")
        return model

    def learn(
        self,
        total_timesteps: int,
        callback=None,
        progress_bar: bool = False,
        reset_num_timesteps: bool = True,
    ) -> None:
        del progress_bar
        self.last_reset_num_timesteps = reset_num_timesteps
        if callback is not None:
            callback.locals = {
                "rewards": np.array([0.5], dtype=np.float32),
                "dones": np.array([False]),
            }
            callback.num_timesteps = total_timesteps - 1
            callback._on_step()

            callback.locals = {
                "rewards": np.array([1.5], dtype=np.float32),
                "dones": np.array([True]),
            }
            callback.num_timesteps = total_timesteps
            callback._on_step()

    def save(self, filepath: str) -> None:
        self.saved_path = filepath
        output = (
            Path(f"{filepath}.zip") if not filepath.endswith(".zip") else Path(filepath)
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"fake-model")

    def predict(self, observation, deterministic: bool = True):
        del observation
        del deterministic
        return np.array([1]), {"latent": np.array([0.1], dtype=np.float32)}


def test_trainer_instantiation_and_model_type() -> None:
    """Trainer exposes expected algorithm-driven model type."""

    trainer = StableBaselines3Trainer(algorithm=SB3Algorithm.PPO, policy="MlpPolicy")

    assert trainer.model_type == "ppo"
    assert trainer.is_trained is False


def test_create_model_uses_algorithm_mapping_and_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """create_model builds model with expected arguments from TrainingConfig."""

    trainer = StableBaselines3Trainer(algorithm=SB3Algorithm.PPO, policy="MlpPolicy")
    monkeypatch.setitem(trainer._MODEL_CLASS_MAP, SB3Algorithm.PPO, _FakeModel)

    config = TrainingConfig(
        total_timesteps=10,
        learning_rate=0.0001,
        batch_size=32,
        n_steps=64,
        tensorboard_log="/tmp/tb",
        custom_params={"gamma": 0.99},
    )

    trainer.create_model(_DummyEnv(), config)

    assert isinstance(trainer._model, _FakeModel)
    assert trainer._model.kwargs["learning_rate"] == pytest.approx(0.0001)
    assert trainer._model.kwargs["batch_size"] == 32
    assert trainer._model.kwargs["n_steps"] == 64
    assert trainer._model.kwargs["gamma"] == pytest.approx(0.99)


def test_train_provides_progress_callback_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train forwards progress updates and returns expected aggregate result."""

    trainer = StableBaselines3Trainer(algorithm=SB3Algorithm.PPO)
    monkeypatch.setitem(trainer._MODEL_CLASS_MAP, SB3Algorithm.PPO, _FakeModel)
    trainer.create_model(_DummyEnv(), TrainingConfig(total_timesteps=12))

    events: list[dict[str, object]] = []
    result = trainer.train(TrainingConfig(total_timesteps=12), callback=events.append)

    assert trainer.is_trained is True
    assert result.timesteps_trained == 12
    assert result.episodes_completed == 1
    assert result.final_reward == pytest.approx(1.5)
    assert len(events) == 2
    assert events[-1]["timesteps"] == 12


def test_train_forwards_reset_num_timesteps_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train forwards reset_num_timesteps to learn for warm-start continuation."""

    trainer = StableBaselines3Trainer(algorithm=SB3Algorithm.PPO)
    monkeypatch.setitem(trainer._MODEL_CLASS_MAP, SB3Algorithm.PPO, _FakeModel)
    trainer.create_model(_DummyEnv(), TrainingConfig(total_timesteps=12))

    trainer.train(TrainingConfig(total_timesteps=12))
    assert trainer._model.last_reset_num_timesteps is True

    trainer.train(TrainingConfig(total_timesteps=12), reset_num_timesteps=False)
    assert trainer._model.last_reset_num_timesteps is False


def test_predict_raises_when_model_not_ready() -> None:
    """predict enforces ensure_trained contract."""

    trainer = StableBaselines3Trainer(algorithm=SB3Algorithm.PPO)

    with pytest.raises(ModelNotTrainedError):
        trainer.predict({"price": np.array([1.0], dtype=np.float32)})


def test_evaluate_returns_metrics_with_results_dataframe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """evaluate runs episodes and returns summary metrics and detailed results."""

    trainer = StableBaselines3Trainer(algorithm=SB3Algorithm.PPO)
    monkeypatch.setitem(trainer._MODEL_CLASS_MAP, SB3Algorithm.PPO, _FakeModel)
    trainer.create_model(_DummyEnv(), TrainingConfig(total_timesteps=5))
    trainer.train(TrainingConfig(total_timesteps=5))

    result = trainer.evaluate(_DummyEnv(), n_episodes=2)

    assert result.episodes == 2
    assert result.mean_reward == pytest.approx(3.0)
    assert result.mean_episode_length == pytest.approx(3.0)
    assert result.results_df is not None
    assert len(result.results_df) == 2


def test_save_and_load_dqn_replay_buffer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """DQN trainer persists and restores replay buffer sidecar file."""

    trainer = StableBaselines3Trainer(algorithm=SB3Algorithm.DQN)
    monkeypatch.setitem(trainer._MODEL_CLASS_MAP, SB3Algorithm.DQN, _FakeModel)
    trainer.create_model(_DummyEnv(), TrainingConfig(total_timesteps=5))
    trainer.train(TrainingConfig(total_timesteps=5))

    save_path = tmp_path / "model_artifacts" / "dqn_model"
    trainer.save(str(save_path))

    replay_buffer_file = Path(f"{save_path}_replay_buffer.pkl")
    assert replay_buffer_file.exists()

    loaded = StableBaselines3Trainer(algorithm=SB3Algorithm.DQN)
    monkeypatch.setitem(loaded._MODEL_CLASS_MAP, SB3Algorithm.DQN, _FakeModel)
    loaded.load(str(save_path))

    assert loaded.is_trained is True
    assert isinstance(loaded._model, _FakeModel)
    assert isinstance(loaded._model.replay_buffer, _FakeReplayBuffer)
