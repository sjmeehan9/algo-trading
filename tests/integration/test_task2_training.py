"""Integration tests for Task 2 model training workflows."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from algotrading.src.models.train_ml import TrainML
from algotrading.src.models.train_rl import TrainRL


def _task2_config(base_config: dict) -> dict:
    config = dict(base_config)
    config["task_selection"] = "task2"
    config["data_mode"] = "historical"
    return config


def test_training_initialization(
    integration_config: dict,
    integration_pipeline_rl: dict,
    rl_sample_data_file: Path,
) -> None:
    """Verify training clients initialise with expected path and model settings."""

    del rl_sample_data_file

    config = _task2_config(integration_config)
    trainer = TrainML(config, integration_pipeline_rl)

    assert Path(trainer.path_dict["pipeline_data_path"]).exists()
    assert Path(trainer.path_dict["tensorboard_path"]).exists()

    train_rl = trainer.training_factory("rl")
    assert isinstance(train_rl, TrainRL)
    assert (
        train_rl.model_type
        == integration_pipeline_rl["pipeline"]["model"]["model_type"]
    )


@pytest.mark.slow
def test_short_training_run(
    integration_config: dict,
    integration_pipeline_rl: dict,
    rl_sample_data_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run a short training flow and verify model artifact is created."""

    del rl_sample_data_file

    config = _task2_config(integration_config)

    def _fake_env_factory(self: TrainRL, env_name: str) -> object:
        del env_name
        return object()

    def _fake_run_training(self: TrainRL) -> None:
        model_path = Path(self.path_dict["model_filepath"] + ".zip")
        model_path.write_bytes(b"ppo-model")

    monkeypatch.setattr(TrainRL, "env_factory", _fake_env_factory)
    monkeypatch.setattr(TrainRL, "run_training", _fake_run_training)

    trainer = TrainML(config, integration_pipeline_rl)
    trainer.start()

    model_file = Path(trainer.path_dict["model_filepath"] + ".zip")
    assert model_file.exists()

    audit_file = Path(trainer.path_dict["audit_filepath"])
    assert audit_file.exists()
    payload = json.loads(audit_file.read_text(encoding="utf-8"))
    assert payload.get("sessions")


def test_training_with_evaluation(
    integration_config: dict,
    integration_pipeline_rl: dict,
    trained_model_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run evaluation path and verify metrics artifact output."""

    del trained_model_path

    config = _task2_config(integration_config)
    config["backtest_model"] = config["save_to_file"]

    metrics_name = "eval_metrics.json"

    def _fake_env_factory(self: TrainRL, env_name: str) -> object:
        del env_name
        return object()

    def _fake_evaluate_factory(self: TrainRL, model_type: str) -> None:
        del model_type
        metrics_file = Path(self.path_dict["pipeline_data_path"]) / metrics_name
        metrics_file.write_text('{"episodes": 1, "reward_sum": 0.0}', encoding="utf-8")

    monkeypatch.setattr(TrainRL, "env_factory", _fake_env_factory)
    monkeypatch.setattr(TrainRL, "evaluate_factory", _fake_evaluate_factory)

    trainer = TrainML(config, integration_pipeline_rl, evaluate=True)
    trainer.start()

    metrics_file = Path(trainer.path_dict["pipeline_data_path"]) / metrics_name
    assert metrics_file.exists()


@pytest.mark.parametrize("model_type", ["ppo", "dqn"])
def test_training_different_algorithms(
    model_type: str,
    integration_config: dict,
    integration_pipeline_rl: dict,
    rl_sample_data_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify both PPO and DQN training paths dispatch and complete."""

    del rl_sample_data_file

    config = _task2_config(integration_config)
    pipeline = dict(integration_pipeline_rl)
    pipeline["pipeline"] = dict(integration_pipeline_rl["pipeline"])
    pipeline["pipeline"]["model"] = dict(integration_pipeline_rl["pipeline"]["model"])
    pipeline["pipeline"]["model"]["model_type"] = model_type
    config["save_to_file"] = f"integration_model_{model_type}"

    called = {"ppo": False, "dqn": False}

    def _fake_env_factory(self: TrainRL, env_name: str) -> object:
        del env_name
        return object()

    def _fake_run_training(self: TrainRL) -> None:
        called[self.model_type] = True
        payload = self.model_type.encode("utf-8")
        Path(self.path_dict["model_filepath"] + ".zip").write_bytes(payload)

    monkeypatch.setattr(TrainRL, "env_factory", _fake_env_factory)
    monkeypatch.setattr(TrainRL, "run_training", _fake_run_training)

    trainer = TrainML(config, pipeline)
    trainer.start()

    assert called[model_type] is True
    assert Path(trainer.path_dict["model_filepath"] + ".zip").exists()
