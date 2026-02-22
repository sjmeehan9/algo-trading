"""Integration tests for RL trainer implementations and trainer wiring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from algotrading.src.models.train_ml import TrainML
from algotrading.src.models.train_rl import TrainRL
from algotrading.src.trainers import (
    SB3Algorithm,
    StableBaselines3Trainer,
    TrainingConfig,
)
from gymnasium import Env, spaces

from tests.mocks import MockRLTrainer


class _TinyEnv(Env):
    """Small deterministic environment for short trainer integration checks."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        self.observation_space = spaces.Dict(
            {
                "price": spaces.Box(
                    low=-10.0,
                    high=10.0,
                    shape=(4,),
                    dtype=np.float32,
                )
            }
        )
        self.action_space = spaces.Discrete(3)
        self._step = 0

    def reset(self, *, seed=None, options=None):  # type: ignore[override]
        super().reset(seed=seed)
        del options
        self._step = 0
        return {"price": np.zeros(4, dtype=np.float32)}, {}

    def step(self, action: int):  # type: ignore[override]
        self._step += 1
        obs = {"price": np.full(4, self._step * 0.1, dtype=np.float32)}
        reward = 1.0 - abs(float(action - 1)) * 0.1
        terminated = self._step >= 10
        truncated = False
        info = {"step": self._step}
        return obs, reward, terminated, truncated, info


@pytest.mark.slow
def test_sb3_trainer_ppo_creation() -> None:
    """PPO trainer should create model for valid gym environment."""

    trainer = StableBaselines3Trainer(
        algorithm=SB3Algorithm.PPO, policy="MultiInputPolicy"
    )
    trainer.create_model(
        _TinyEnv(),
        TrainingConfig(
            total_timesteps=64, n_steps=32, batch_size=32, custom_params={"verbose": 0}
        ),
    )
    assert trainer.model_type == "ppo"


@pytest.mark.slow
def test_sb3_trainer_dqn_creation() -> None:
    """DQN trainer should create model for valid gym environment."""

    trainer = StableBaselines3Trainer(
        algorithm=SB3Algorithm.DQN, policy="MultiInputPolicy"
    )
    trainer.create_model(
        _TinyEnv(),
        TrainingConfig(
            total_timesteps=64,
            batch_size=32,
            custom_params={
                "verbose": 0,
                "buffer_size": 512,
                "learning_starts": 0,
                "train_freq": 1,
                "gradient_steps": 1,
            },
        ),
    )
    assert trainer.model_type == "dqn"


@pytest.mark.slow
def test_sb3_trainer_short_training() -> None:
    """Short SB3 training run should return a populated TrainingResult."""

    trainer = StableBaselines3Trainer(
        algorithm=SB3Algorithm.PPO, policy="MultiInputPolicy"
    )
    env = _TinyEnv()
    config = TrainingConfig(
        total_timesteps=128,
        learning_rate=0.0003,
        batch_size=32,
        n_steps=32,
        custom_params={"verbose": 0},
    )
    trainer.create_model(env, config)
    result = trainer.train(config)
    assert result.timesteps_trained == 128
    assert trainer.is_trained is True


@pytest.mark.slow
def test_sb3_trainer_save_load(tmp_path: Path) -> None:
    """Trained SB3 model should round-trip via save/load and predict."""

    env = _TinyEnv()
    trainer = StableBaselines3Trainer(
        algorithm=SB3Algorithm.PPO, policy="MultiInputPolicy"
    )
    config = TrainingConfig(
        total_timesteps=96, n_steps=32, batch_size=32, custom_params={"verbose": 0}
    )

    trainer.create_model(env, config)
    trainer.train(config)

    model_base = tmp_path / "sb3_model"
    trainer.save(str(model_base))

    restored = StableBaselines3Trainer(
        algorithm=SB3Algorithm.PPO, policy="MultiInputPolicy"
    )
    restored.load(str(model_base), env=env)

    obs, _ = env.reset()
    action, _ = restored.predict(obs)
    assert action in (0, 1, 2)


def test_mock_trainer_predict() -> None:
    """Mock trainer should return configured action without SB3 dependency."""

    trainer = MockRLTrainer()
    trainer.set_predict_action(2)
    trainer.load("unused")

    action, info = trainer.predict({"price": [1, 2, 3, 4]})
    assert action == 2
    assert "states" in info


def test_trainrl_with_mock_trainer(
    integration_config: dict,
    integration_pipeline_rl: dict,
    rl_sample_data_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TrainML/TrainRL should complete end-to-end with injected mock trainer."""

    del rl_sample_data_file
    config = dict(integration_config)
    config["task_selection"] = "task2"

    trainer = MockRLTrainer()

    def _fake_env_factory(self: TrainRL, env_name: str) -> object:
        del env_name
        return object()

    monkeypatch.setattr(TrainRL, "env_factory", _fake_env_factory)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")

    train_ml = TrainML(config, integration_pipeline_rl, trainer=trainer)
    train_ml.start()

    model_file = Path(train_ml.path_dict["model_filepath"])
    assert model_file.exists()
    assert trainer.train_calls == 1
