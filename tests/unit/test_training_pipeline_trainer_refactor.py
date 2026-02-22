"""Unit tests for training pipeline trainer-interface refactor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from algotrading.src.models.predict import Predict
from algotrading.src.models.train_ml import TrainML
from algotrading.src.models.train_rl import TrainRL
from algotrading.src.models.trainer_factory import create_rl_trainer
from algotrading.src.trainers import (
    EvaluationResult,
    RLTrainer,
    TrainingConfig,
    TrainingResult,
)


class _MockRLTrainer(RLTrainer):
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self._is_trained = False

    def create_model(self, env, config: TrainingConfig) -> None:
        self.calls.append(("create_model", config))

    def train(self, config: TrainingConfig, callback=None) -> TrainingResult:
        del callback
        self.calls.append(("train", config))
        self._is_trained = True
        return TrainingResult(
            timesteps_trained=config.total_timesteps,
            episodes_completed=1,
            final_reward=0.0,
            training_time_seconds=0.1,
        )

    def evaluate(self, env, n_episodes: int = 10) -> EvaluationResult:
        del env, n_episodes
        self.calls.append(("evaluate", None))
        return EvaluationResult(
            episodes=1,
            mean_reward=0.0,
            std_reward=0.0,
            mean_episode_length=1.0,
        )

    def save(self, filepath: str) -> None:
        self.calls.append(("save", filepath))

    def load(self, filepath: str, env=None) -> None:
        del env
        self.calls.append(("load", filepath))
        self._is_trained = True

    def predict(self, observation: dict[str, object], deterministic: bool = True):
        del observation, deterministic
        self.calls.append(("predict", None))
        return 1, {"states": None}

    @property
    def model_type(self) -> str:
        return "ppo"

    @property
    def is_trained(self) -> bool:
        return self._is_trained


def test_create_rl_trainer_maps_supported_algorithms(
    mock_pipeline_config: dict,
) -> None:
    """Factory should create SB3 trainer for PPO and DQN model types."""

    pipeline = {"pipeline": dict(mock_pipeline_config["pipeline"])}
    trainer = create_rl_trainer(pipeline)
    assert trainer.model_type == "ppo"

    pipeline["pipeline"]["model"] = dict(mock_pipeline_config["pipeline"]["model"])
    pipeline["pipeline"]["model"]["model_type"] = "dqn"
    trainer = create_rl_trainer(pipeline)
    assert trainer.model_type == "dqn"


def test_trainrl_run_training_delegates_to_trainer(
    mock_config: dict,
    mock_pipeline_config: dict,
    tmp_path: Path,
) -> None:
    """TrainRL should delegate model lifecycle operations to injected trainer."""

    config = dict(mock_config)
    config["input_model"] = "missing_input"

    path_dict = {
        "input_filepath": str(tmp_path / "missing_input.zip"),
        "tensorboard_path": str(tmp_path / "tensorboard"),
        "model_filepath": str(tmp_path / "model_out"),
    }

    trainer = _MockRLTrainer()
    train_rl = TrainRL(
        config, mock_pipeline_config, path_dict, evaluate=False, trainer=trainer
    )

    @dataclass
    class _StateBuilderStub:
        total_timesteps: int = 128

    train_rl.state_builder = _StateBuilderStub()
    train_rl.env = object()

    train_rl.run_training()

    call_names = [entry[0] for entry in trainer.calls]
    assert call_names == ["create_model", "train", "save"]


def test_predict_uses_injected_trainer_for_load_and_inference(
    mock_config: dict,
    mock_pipeline_config: dict,
    tmp_path: Path,
) -> None:
    """Predict should call trainer load on init and trainer predict for inference."""

    config = dict(mock_config)
    config["data_path"] = str(tmp_path) + "/"
    config["trading_model"] = "model_x"

    pipeline = {"pipeline": dict(mock_pipeline_config["pipeline"])}
    pipeline["pipeline"]["filename"] = "pipeline_x"
    pipeline["pipeline"]["pipeline_type"] = "rl"

    trainer = _MockRLTrainer()
    predictor = Predict(config, pipeline, trainer=trainer)

    action, _states = predictor.get_action({"foo": 1})

    assert action == 1
    call_names = [entry[0] for entry in trainer.calls]
    assert "load" in call_names
    assert "predict" in call_names


def test_trainml_passes_trainer_to_trainrl(
    mock_config: dict,
    mock_pipeline_config: dict,
    tmp_path: Path,
) -> None:
    """TrainML should pass injected trainer through training factory."""

    config = dict(mock_config)
    config["data_path"] = str(tmp_path) + "/"
    trainer = _MockRLTrainer()

    train_ml = TrainML(config, mock_pipeline_config, evaluate=False, trainer=trainer)
    train_rl = train_ml.training_factory("rl")

    assert isinstance(train_rl, TrainRL)
    assert train_rl.trainer is trainer
