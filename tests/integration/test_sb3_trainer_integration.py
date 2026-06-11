"""Integration tests for StableBaselines3Trainer using real SB3 training loops."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from algotrading.src.trainers import (
    SB3Algorithm,
    StableBaselines3Trainer,
    TrainingConfig,
)
from gymnasium import Env, spaces


class _TinyTradingEnv(Env):
    """Small deterministic environment for short integration training runs."""

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
        self._step = 0
        obs = {"price": np.zeros(4, dtype=np.float32)}
        info: dict[str, object] = {}
        return obs, info

    def step(self, action: int):  # type: ignore[override]
        self._step += 1
        direction = float(action - 1)
        obs = {"price": np.full(4, fill_value=self._step * 0.1, dtype=np.float32)}
        reward = 1.0 - abs(direction) * 0.1
        terminated = self._step >= 12
        truncated = False
        info: dict[str, object] = {"step": self._step}
        return obs, reward, terminated, truncated, info


@pytest.mark.slow
def test_short_ppo_training_and_prediction() -> None:
    """PPO trainer can create, train, and predict on a tiny environment."""

    env = _TinyTradingEnv()
    trainer = StableBaselines3Trainer(
        algorithm=SB3Algorithm.PPO, policy="MultiInputPolicy"
    )

    config = TrainingConfig(
        total_timesteps=128,
        learning_rate=0.0003,
        batch_size=32,
        n_steps=32,
        custom_params={"verbose": 0},
    )

    trainer.create_model(env, config)
    result = trainer.train(config)

    obs, _ = env.reset()
    action, info = trainer.predict(obs)

    assert result.timesteps_trained == 128
    assert trainer.is_trained is True
    assert action in (0, 1, 2)
    assert "states" in info


@pytest.mark.slow
def test_short_dqn_training_save_load_with_replay_buffer(tmp_path: Path) -> None:
    """DQN trainer saves and restores both model artifact and replay buffer."""

    env = _TinyTradingEnv()
    trainer = StableBaselines3Trainer(
        algorithm=SB3Algorithm.DQN, policy="MultiInputPolicy"
    )

    config = TrainingConfig(
        total_timesteps=160,
        learning_rate=0.0003,
        batch_size=32,
        custom_params={
            "verbose": 0,
            "buffer_size": 512,
            "learning_starts": 0,
            "train_freq": 1,
            "gradient_steps": 1,
        },
    )

    trainer.create_model(env, config)
    trainer.train(config)

    model_base = tmp_path / "dqn_artifacts" / "model"
    trainer.save(str(model_base))

    assert Path(f"{model_base}.zip").exists()
    assert Path(f"{model_base}_replay_buffer.pkl").exists()

    restored = StableBaselines3Trainer(
        algorithm=SB3Algorithm.DQN, policy="MultiInputPolicy"
    )
    restored.load(str(model_base), env=env)

    obs, _ = env.reset()
    action, _ = restored.predict(obs)
    assert action in (0, 1, 2)


@pytest.mark.slow
def test_continue_training_warm_starts_from_saved_artifact(tmp_path: Path) -> None:
    """Loading an artifact and training with reset_num_timesteps=False continues."""

    env = _TinyTradingEnv()
    trainer = StableBaselines3Trainer(
        algorithm=SB3Algorithm.PPO, policy="MultiInputPolicy"
    )
    config = TrainingConfig(
        total_timesteps=128,
        learning_rate=0.0003,
        batch_size=32,
        n_steps=32,
        custom_params={"verbose": 0},
    )
    trainer.create_model(env, config)
    trainer.train(config)
    model_base = tmp_path / "ppo_artifacts" / "model"
    trainer.save(str(model_base))

    # A fresh trainer continues from the saved weights instead of re-initialising.
    continued = StableBaselines3Trainer(
        algorithm=SB3Algorithm.PPO, policy="MultiInputPolicy"
    )
    continued.load(str(model_base), env=env)
    assert continued._model is not None
    timesteps_before = continued._model.num_timesteps
    assert timesteps_before >= 128

    continued.train(config, reset_num_timesteps=False)

    # The counter accumulated rather than resetting, proving warm-start.
    assert continued._model.num_timesteps > timesteps_before


@pytest.mark.slow
def test_evaluate_returns_non_empty_metrics() -> None:
    """evaluate returns aggregate metrics and per-episode detail dataframe."""

    env = _TinyTradingEnv()
    trainer = StableBaselines3Trainer(
        algorithm=SB3Algorithm.PPO, policy="MultiInputPolicy"
    )

    config = TrainingConfig(
        total_timesteps=96,
        batch_size=32,
        n_steps=32,
        custom_params={"verbose": 0},
    )

    trainer.create_model(env, config)
    trainer.train(config)
    evaluation = trainer.evaluate(_TinyTradingEnv(), n_episodes=3)

    assert evaluation.episodes == 3
    assert evaluation.results_df is not None
    assert not evaluation.results_df.empty
