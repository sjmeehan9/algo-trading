"""Integration tests for Phase 5.11 deployment selection API endpoints."""

from __future__ import annotations

from pathlib import Path

from algotrading.api.config import APIConfig
from algotrading.api.main import create_app
from algotrading.api.schemas.models import ModelConfigCreate, ModelType
from algotrading.api.services import ModelService
from algotrading.api.services.deployment_service import DeploymentService
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
from fastapi.testclient import TestClient


def _auth_headers() -> dict[str, str]:
    """Return API authentication headers."""

    return {"X-API-Key": "deployment-secret-key"}


def _build_model_service(tmp_path: Path) -> ModelService:
    """Create an isolated model service for deployment API tests."""

    return ModelService(
        supporting_registry=SupportingModelRegistry(),
        strategy_registry=CustomStrategyRegistry(strategy_dirs=[], auto_scan=False),
        generation_tracker=GenerationTracker(
            JsonFileStorage(str(tmp_path / "generations"))
        ),
        core_models_path=tmp_path / "core_models.json",
    )


def _create_core_model(service: ModelService, name: str = "Deployable Core") -> str:
    """Create a core RL model and return its ID."""

    model = service.create_model(
        ModelConfigCreate(
            name=name,
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
            name="Supporting Sentiment",
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


def _add_evaluated_generation(service: ModelService, model_id: str) -> str:
    """Add an evaluated generation for deployment API tests."""

    generation = service.generation_tracker.start_generation(
        model_id=model_id,
        hyperparameters={"learning_rate": 0.0003},
    )
    service.generation_tracker.complete_generation(
        generation.generation_id,
        training_metrics=TrainingMetrics(
            final_reward=2.0,
            mean_reward=1.6,
            std_reward=0.2,
            episodes_completed=9,
            timesteps_trained=9000,
            training_time_seconds=180.0,
        ),
        model_path="models/api-deployment/model.zip",
    )
    service.generation_tracker.add_evaluation(
        generation.generation_id,
        EvaluationMetrics(
            sharpe_ratio=1.3,
            max_drawdown=0.05,
            total_return=0.14,
            win_rate=0.58,
            profit_factor=1.8,
            num_trades=18,
        ),
    )
    return generation.generation_id


def _api_client(tmp_path: Path) -> tuple[TestClient, ModelService]:
    """Build a configured deployment API test client."""

    app = create_app(APIConfig(api_key="deployment-secret-key", debug=True))
    model_service = _build_model_service(tmp_path)
    app.state.model_service = model_service
    app.state.deployment_service = DeploymentService(
        model_service=model_service,
        selection_path=tmp_path / "deployment_selection.json",
    )
    return TestClient(app), model_service


def test_candidates_endpoint_returns_only_core_rl_models(tmp_path: Path) -> None:
    """Deployment candidates should exclude supporting model roots."""

    client, model_service = _api_client(tmp_path)
    core_model_id = _create_core_model(model_service)
    _create_supporting_model(model_service)

    response = client.get("/api/v1/deployment/candidates", headers=_auth_headers())

    assert response.status_code == 200, response.text
    candidates = response.json()["data"]
    assert [candidate["model_id"] for candidate in candidates] == [core_model_id]


def test_validation_rejects_supporting_models_as_roots(tmp_path: Path) -> None:
    """Supporting models should never be valid deployment root candidates."""

    client, model_service = _api_client(tmp_path)
    supporting_model_id = _create_supporting_model(model_service)

    response = client.post(
        "/api/v1/deployment/validate",
        headers=_auth_headers(),
        json={"model_id": supporting_model_id, "generation_id": "any"},
    )

    assert response.status_code == 400
    assert response.json()["error_code"] == "DEPLOYMENT_VALIDATION_ERROR"


def test_persist_and_retrieve_valid_deployment_selection(tmp_path: Path) -> None:
    """A valid deployment candidate should save and reload through the API."""

    client, model_service = _api_client(tmp_path)
    model_id = _create_core_model(model_service)
    generation_id = _add_evaluated_generation(model_service, model_id)

    save_response = client.put(
        "/api/v1/deployment/selection",
        headers=_auth_headers(),
        json={"model_id": model_id, "generation_id": generation_id},
    )
    assert save_response.status_code == 200, save_response.text
    saved = save_response.json()["data"]
    assert saved["model_id"] == model_id
    assert saved["generation_id"] == generation_id
    assert saved["readiness"]["deployable"] is True

    get_response = client.get("/api/v1/deployment/selection", headers=_auth_headers())
    assert get_response.status_code == 200
    assert get_response.json()["data"]["generation_id"] == generation_id

    clear_response = client.delete(
        "/api/v1/deployment/selection", headers=_auth_headers()
    )
    assert clear_response.status_code == 200

    empty_response = client.get("/api/v1/deployment/selection", headers=_auth_headers())
    assert empty_response.status_code == 200
    assert empty_response.json()["data"] is None
