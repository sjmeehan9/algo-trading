"""Unit tests for pre-deployment selection service behavior."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from algotrading.api.schemas.deployment import DeploymentValidationRequest
from algotrading.api.schemas.models import ModelConfigCreate, ModelType
from algotrading.api.services import GenerationNotFoundError, ModelNotFoundError
from algotrading.api.services.deployment_service import (
    DeploymentSelectionNotFoundError,
    DeploymentService,
    DeploymentValidationError,
)
from algotrading.api.services.model_service import ModelService
from algotrading.src.models.registry import (
    CustomStrategyRegistry,
    SupportingModelRegistry,
)
from algotrading.src.models.signals import SignalType
from algotrading.src.models.tracking import (
    EvaluationMetrics,
    GenerationTracker,
    JsonFileStorage,
    TrainingMetrics,
)


@pytest.fixture()
def model_service(tmp_path: Path) -> ModelService:
    """Create an isolated model service for deployment tests."""

    return ModelService(
        supporting_registry=SupportingModelRegistry(),
        strategy_registry=CustomStrategyRegistry(strategy_dirs=[], auto_scan=False),
        generation_tracker=GenerationTracker(
            JsonFileStorage(str(tmp_path / "generations"))
        ),
        core_models_path=tmp_path / "core_models.json",
    )


def _create_core_model(service: ModelService) -> str:
    """Create a baseline core RL model and return its ID."""

    model = service.create_model(
        ModelConfigCreate(
            name="Deployment Core",
            model_type=ModelType.CORE_RL,
            trainer_type="stable_baselines3",
            algorithm="ppo",
            hyperparameters={"learning_rate": 0.0003, "n_steps": 2048},
            training_data_config={"symbols": ["AAPL"]},
            supporting_model_ids=[],
            strategy_ids=[],
            environment_config={"initial_capital": 100000},
            reward_function="profit_seeker",
        )
    )
    return model.model_id


def _create_supporting_model(service: ModelService) -> str:
    """Create a supporting model and return its ID."""

    model = service.create_model(
        ModelConfigCreate(
            name="Sentiment Helper",
            model_type=ModelType.SUPPORTING_ML,
            signal_type=SignalType.SENTIMENT,
            trainer_type="sklearn",
            algorithm="random_forest",
            hyperparameters={"n_estimators": 100},
            input_data_types=["news_text"],
            input_frequency="1m",
        )
    )
    return model.model_id


def _add_generation(
    service: ModelService,
    model_id: str,
    *,
    evaluated: bool,
) -> str:
    """Add a completed generation, optionally with evaluation metrics."""

    generation = service.generation_tracker.start_generation(
        model_id=model_id,
        hyperparameters={"learning_rate": 0.0003, "n_steps": 2048},
    )
    service.generation_tracker.complete_generation(
        generation.generation_id,
        training_metrics=TrainingMetrics(
            final_reward=1.4,
            mean_reward=1.1,
            std_reward=0.2,
            episodes_completed=8,
            timesteps_trained=8000,
            training_time_seconds=120.0,
        ),
        model_path="models/deployment-core/model.zip",
    )
    if evaluated:
        service.generation_tracker.add_evaluation(
            generation.generation_id,
            EvaluationMetrics(
                sharpe_ratio=1.2,
                max_drawdown=0.04,
                total_return=0.1,
                win_rate=0.55,
                profit_factor=1.6,
                num_trades=12,
            ),
        )
    return generation.generation_id


def test_validate_rejects_missing_model(model_service: ModelService) -> None:
    """Unknown model IDs should raise the model service not-found error."""

    service = DeploymentService(model_service=model_service)

    with pytest.raises(ModelNotFoundError):
        service.validate_selection(
            DeploymentValidationRequest(model_id="missing", generation_id="gen")
        )


def test_validate_rejects_supporting_model_root(model_service: ModelService) -> None:
    """Only core RL models can be used as deployment roots."""

    supporting_model_id = _create_supporting_model(model_service)
    service = DeploymentService(model_service=model_service)

    with pytest.raises(DeploymentValidationError, match="Only core RL models"):
        service.validate_selection(
            DeploymentValidationRequest(
                model_id=supporting_model_id,
                generation_id="unused-generation",
            )
        )


def test_validate_rejects_missing_generation(model_service: ModelService) -> None:
    """Unknown generation IDs should raise a generation not-found error."""

    model_id = _create_core_model(model_service)
    service = DeploymentService(model_service=model_service)

    with pytest.raises(GenerationNotFoundError):
        service.validate_selection(
            DeploymentValidationRequest(
                model_id=model_id,
                generation_id="missing-generation",
            )
        )


def test_not_evaluated_generation_is_not_deployable(
    model_service: ModelService,
) -> None:
    """Completed generations without evaluation evidence should be blocked."""

    model_id = _create_core_model(model_service)
    generation_id = _add_generation(model_service, model_id, evaluated=False)
    service = DeploymentService(model_service=model_service)

    readiness = service.validate_selection(
        DeploymentValidationRequest(model_id=model_id, generation_id=generation_id)
    )

    assert readiness.deployable is False
    assert any("backtest or evaluation" in error for error in readiness.errors)


def test_selection_persistence_create_read_clear(
    model_service: ModelService,
    tmp_path: Path,
) -> None:
    """Valid selections should persist, reload, and clear cleanly."""

    model_id = _create_core_model(model_service)
    generation_id = _add_generation(model_service, model_id, evaluated=True)
    selection_path = tmp_path / "deployment_selection.json"
    service = DeploymentService(
        model_service=model_service,
        selection_path=selection_path,
    )

    selection = service.save_selection(
        DeploymentValidationRequest(model_id=model_id, generation_id=generation_id),
        selected_by="unit-test",
    )

    assert selection.model_id == model_id
    assert selection.generation_id == generation_id
    assert selection.selected_by == "unit-test"
    assert selection.readiness.deployable is True
    assert selection.selected_at <= datetime.now(tz=UTC)
    assert selection_path.exists()

    reloaded = DeploymentService(
        model_service=model_service,
        selection_path=selection_path,
    )
    assert reloaded.get_selection().generation_id == generation_id

    reloaded.clear_selection()
    with pytest.raises(DeploymentSelectionNotFoundError):
        reloaded.get_selection()
