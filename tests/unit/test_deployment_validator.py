"""Unit tests for Phase 6 RL-only deployment enforcement."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from algotrading.api.schemas.models import ModelConfigCreate
from algotrading.api.schemas.models import ModelType as APIModelType
from algotrading.api.services.model_service import ModelService
from algotrading.src.models.registry import (
    CustomStrategyRegistry,
    SupportingModelRegistry,
)
from algotrading.src.models.signals import SignalType
from algotrading.src.models.tracking import (
    GenerationTracker,
    JsonFileStorage,
    TrainingMetrics,
)
from algotrading.src.trading.deployment import (
    DeploymentAuditLog,
    DeploymentValidator,
    InvalidDeploymentError,
    validate_core_model_instance,
    validate_pipeline_deployment_config,
)
from algotrading.src.trainers import (
    EvaluationResult,
    MLPrediction,
    MLTrainer,
    MLTrainingConfig,
    MLTrainingResult,
    RLTrainer,
    TrainingConfig,
    TrainingResult,
)
from gymnasium import Env


@dataclass(slots=True)
class _FakeModel:
    """Minimal model record for validator edge-case tests."""

    model_id: str
    name: str
    model_type: str


class _FakeModelService:
    """Minimal service for model-type validation without API storage."""

    def __init__(self, model: _FakeModel) -> None:
        self.model = model

    def get_model(self, model_id: str) -> _FakeModel:
        """Return the configured fake model."""

        if model_id != self.model.model_id:
            raise KeyError(model_id)
        return self.model

    def list_models(self, page: int = 1, page_size: int = 100) -> object:
        """Return a single-page fake model response."""

        del page, page_size
        return type("Response", (), {"items": [self.model], "pages": 1})()


class _RuntimeRLTrainer(RLTrainer):
    """Trained RL trainer used for runtime validation tests."""

    def create_model(self, env: Env, config: TrainingConfig) -> None:
        """Accept model creation inputs for interface compliance."""

        del env, config

    def train(
        self,
        config: TrainingConfig,
        callback: Callable[[dict[str, object]], None] | None = None,
    ) -> TrainingResult:
        """Return deterministic training metadata."""

        del config, callback
        return TrainingResult(0, 0, 0.0, 0.0)

    def evaluate(self, env: Env, n_episodes: int = 10) -> EvaluationResult:
        """Return deterministic evaluation metadata."""

        del env, n_episodes
        return EvaluationResult(0, 0.0, 0.0, 0.0)

    def save(self, filepath: str) -> None:
        """Accept a save path for interface compliance."""

        del filepath

    def load(self, filepath: str, env: Env | None = None) -> None:
        """Accept a load path for interface compliance."""

        del filepath, env

    def predict(
        self,
        observation: dict[str, object],
        deterministic: bool = True,
    ) -> tuple[int, dict[str, object]]:
        """Return a deterministic hold action."""

        del observation, deterministic
        return 0, {"confidence": 1.0}

    @property
    def model_type(self) -> str:
        """Return an RL model identifier."""

        return "test_rl"

    @property
    def is_trained(self) -> bool:
        """Return that the runtime model is trained."""

        return True


class _RuntimeMLTrainer(MLTrainer):
    """ML trainer used to verify runtime core-model rejection."""

    def create_model(
        self,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        config: MLTrainingConfig,
    ) -> None:
        """Accept model creation inputs for interface compliance."""

        del input_shape, output_shape, config

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: MLTrainingConfig,
    ) -> MLTrainingResult:
        """Return deterministic ML training metadata."""

        del X, y, config
        return MLTrainingResult(
            epochs_trained=0,
            final_loss=0.0,
            validation_loss=None,
            training_time_seconds=0.0,
        )

    def predict(self, X: np.ndarray) -> MLPrediction | list[MLPrediction]:
        """Return a deterministic ML prediction."""

        del X
        return MLPrediction(value=0.0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return deterministic class probabilities."""

        del X
        return np.array([[1.0]])

    def save(self, filepath: str) -> None:
        """Accept a save path for interface compliance."""

        del filepath

    def load(self, filepath: str) -> None:
        """Accept a load path for interface compliance."""

        del filepath

    @property
    def model_type(self) -> str:
        """Return an ML model identifier."""

        return "test_ml"

    @property
    def is_trained(self) -> bool:
        """Return that the runtime model is trained."""

        return True


@pytest.fixture()
def model_service(tmp_path: Path) -> ModelService:
    """Create an isolated model service for validator tests."""

    return ModelService(
        supporting_registry=SupportingModelRegistry(),
        strategy_registry=CustomStrategyRegistry(strategy_dirs=[], auto_scan=False),
        generation_tracker=GenerationTracker(
            JsonFileStorage(str(tmp_path / "generations"))
        ),
        core_models_path=tmp_path / "core_models.json",
    )


def _create_core_model(service: ModelService) -> str:
    """Create a core RL model and return its ID."""

    model = service.create_model(
        ModelConfigCreate(
            name="Deployable Core",
            model_type=APIModelType.CORE_RL,
            trainer_type="stable_baselines3",
            algorithm="ppo",
            hyperparameters={"learning_rate": 0.0003},
            training_data_config={"symbols": ["AAPL"]},
            reward_function="profit_seeker",
        )
    )
    return model.model_id


def _add_completed_generation(service: ModelService, model_id: str) -> str:
    """Add a completed generation with a saved model artifact path."""

    generation = service.generation_tracker.start_generation(
        model_id=model_id,
        hyperparameters={"learning_rate": 0.0003},
    )
    service.generation_tracker.complete_generation(
        generation.generation_id,
        training_metrics=TrainingMetrics(
            final_reward=1.0,
            mean_reward=0.8,
            std_reward=0.1,
            episodes_completed=4,
            timesteps_trained=4000,
            training_time_seconds=60.0,
        ),
        model_path="models/core/model.zip",
    )
    return generation.generation_id


def test_validator_accepts_trained_core_rl_model(
    model_service: ModelService,
    tmp_path: Path,
) -> None:
    """Trained core RL generations should pass the hard deployment gate."""

    model_id = _create_core_model(model_service)
    generation_id = _add_completed_generation(model_service, model_id)
    audit_log = DeploymentAuditLog(tmp_path / "audit.jsonl")
    validator = DeploymentValidator(model_service, audit_log)

    assert validator.validate_deployment(model_id, generation_id=generation_id)
    attempts = audit_log.get_attempts(model_id=model_id)
    assert attempts[-1].result == "approved"
    assert attempts[-1].model_type == "core_rl"


@pytest.mark.parametrize(
    "model_type",
    ["supporting_ml", "supporting_rl", "strategy"],
)
def test_validator_rejects_non_core_model_types(
    model_type: str,
    tmp_path: Path,
) -> None:
    """Supporting models and strategies should be rejected as deployment roots."""

    model = _FakeModel("model-1", "Invalid Root", model_type)
    audit_log = DeploymentAuditLog(tmp_path / f"{model_type}.jsonl")
    validator = DeploymentValidator(_FakeModelService(model), audit_log)

    with pytest.raises(InvalidDeploymentError, match="Only reinforcement learning"):
        validator.validate_deployment("model-1")

    attempts = audit_log.get_attempts(model_id="model-1")
    assert attempts[-1].result == "rejected"
    assert attempts[-1].model_type == model_type


def test_validator_rejects_untrained_core_model(
    model_service: ModelService,
    tmp_path: Path,
) -> None:
    """Core RL models without trained generations should be rejected."""

    model_id = _create_core_model(model_service)
    audit_log = DeploymentAuditLog(tmp_path / "audit.jsonl")
    validator = DeploymentValidator(model_service, audit_log)

    with pytest.raises(InvalidDeploymentError, match="has not been trained"):
        validator.validate_deployment(model_id)

    assert audit_log.get_attempts(model_id=model_id)[-1].result == "rejected"


def test_deployable_models_include_only_trained_core_roots(
    model_service: ModelService,
    tmp_path: Path,
) -> None:
    """Deployable listing should omit untrained and non-core models."""

    trained_id = _create_core_model(model_service)
    generation_id = _add_completed_generation(model_service, trained_id)
    _create_core_model(model_service)
    model_service.create_model(
        ModelConfigCreate(
            name="Sentiment Helper",
            model_type=APIModelType.SUPPORTING_ML,
            signal_type=SignalType.SENTIMENT,
            trainer_type="sklearn",
            algorithm="random_forest",
            hyperparameters={"n_estimators": 100},
            input_data_types=["news_text"],
            input_frequency="1m",
        )
    )

    validator = DeploymentValidator(
        model_service, DeploymentAuditLog(tmp_path / "audit.jsonl")
    )

    deployable = validator.get_deployable_models()

    assert [item["model_id"] for item in deployable] == [trained_id]
    assert deployable[0]["latest_generation"] == generation_id


def test_audit_log_writes_reads_and_filters(tmp_path: Path) -> None:
    """Audit log should persist JSONL entries and filter them by model."""

    audit_log = DeploymentAuditLog(tmp_path / "deployment_audit.jsonl")
    audit_log.log_attempt(
        model_id="core-1",
        result="approved",
        model_type="core_rl",
        generation_id="gen-1",
    )
    audit_log.log_attempt(
        model_id="supporting-1",
        result="rejected",
        reason="Only core RL models can deploy",
        model_type="supporting_ml",
    )

    attempts = audit_log.get_attempts(model_id="core-1")

    assert len(attempts) == 1
    assert attempts[0].result == "approved"
    assert attempts[0].generation_id == "gen-1"


def test_runtime_core_model_instance_validation() -> None:
    """Runtime inference should only accept RLTrainer core models."""

    validate_core_model_instance(_RuntimeRLTrainer(), configured_model_type="core_rl")

    with pytest.raises(InvalidDeploymentError, match="must implement RLTrainer"):
        validate_core_model_instance(
            _RuntimeMLTrainer(), configured_model_type="core_rl"
        )

    with pytest.raises(InvalidDeploymentError, match="supporting_rl"):
        validate_core_model_instance(
            _RuntimeRLTrainer(),
            configured_model_type="supporting_rl",
        )


def test_live_trading_pipeline_config_rejects_strategy_roots() -> None:
    """Legacy live-trading config should block strategy roots."""

    with pytest.raises(InvalidDeploymentError, match="strategy"):
        validate_pipeline_deployment_config({"pipeline": {"pipeline_type": "strategy"}})

    validate_pipeline_deployment_config({"pipeline": {"pipeline_type": "rl"}})
