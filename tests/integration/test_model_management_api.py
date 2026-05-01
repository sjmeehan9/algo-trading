"""Integration tests for Phase 5.2 model management API endpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from algotrading.api.config import APIConfig
from algotrading.api.main import create_app
from algotrading.api.services import ModelService
from algotrading.src.models.registry import (
    CustomStrategyRegistry,
    SupportingModelRegistry,
    trading_strategy,
)
from algotrading.src.models.signals import ModelSignal, SignalMetadata, SignalType
from algotrading.src.models.tracking import (
    EvaluationMetrics,
    GenerationTracker,
    JsonFileStorage,
    TrainingMetrics,
)
from fastapi.testclient import TestClient


@trading_strategy(name="Integration Strategy", signal_type=SignalType.POSITION)
class _IntegrationStrategy:
    """Simple strategy implementation for API integration tests."""

    def predict(self, state: dict[str, object], market_data: object) -> ModelSignal:
        del state, market_data
        return ModelSignal(
            timestamp=datetime.now(tz=UTC),
            signal_type=SignalType.POSITION,
            value=1.0,
            metadata=SignalMetadata(
                model_id="integration_strategy",
                model_type="strategy",
            ),
        )


def _build_service(tmp_path: Path) -> ModelService:
    """Create an isolated service instance for API integration tests."""

    supporting_registry = SupportingModelRegistry()
    strategy_registry = CustomStrategyRegistry(strategy_dirs=[], auto_scan=False)
    strategy_registry.register_strategy(_IntegrationStrategy)
    generation_tracker = GenerationTracker(
        JsonFileStorage(str(tmp_path / "generations"))
    )

    return ModelService(
        supporting_registry=supporting_registry,
        strategy_registry=strategy_registry,
        generation_tracker=generation_tracker,
        core_models_path=tmp_path / "core_models.json",
    )


@pytest.fixture()
def api_client(tmp_path: Path) -> tuple[TestClient, ModelService]:
    """Build a test client with an isolated in-memory-like model service."""

    config = APIConfig(
        api_key="integration-secret-key",
        debug=True,
        cors_origins=["http://localhost:3000"],
    )
    app = create_app(config)
    service = _build_service(tmp_path)
    app.state.model_service = service

    with TestClient(app) as client:
        yield client, service


def _auth_headers() -> dict[str, str]:
    """Return API key headers for authenticated endpoint access."""

    return {"X-API-Key": "integration-secret-key"}


def _core_model_payload(name: str = "Core RL Model") -> dict[str, object]:
    """Return a valid core-model API payload."""

    return {
        "name": name,
        "description": "core rl model",
        "model_type": "core_rl",
        "trainer_type": "stable_baselines3",
        "algorithm": "ppo",
        "hyperparameters": {"learning_rate": 0.0003, "n_steps": 2048},
        "training_data_config": {"symbols": ["AAPL"]},
        "supporting_model_ids": [],
        "strategy_ids": [],
        "environment_config": {"initial_capital": 100000},
        "reward_function": "profit_seeker",
    }


def test_model_crud_flow(api_client: tuple[TestClient, ModelService]) -> None:
    """Create, read, update, and delete operations should work end-to-end."""

    client, _service = api_client

    create_response = client.post(
        "/api/v1/models",
        headers=_auth_headers(),
        json=_core_model_payload(),
    )
    assert create_response.status_code == 201
    created = create_response.json()["data"]
    model_id = created["model_id"]

    get_response = client.get(f"/api/v1/models/{model_id}", headers=_auth_headers())
    assert get_response.status_code == 200
    assert get_response.json()["data"]["model_id"] == model_id

    update_response = client.put(
        f"/api/v1/models/{model_id}",
        headers=_auth_headers(),
        json={"description": "updated description"},
    )
    assert update_response.status_code == 200
    assert update_response.json()["data"]["description"] == "updated description"

    delete_response = client.delete(
        f"/api/v1/models/{model_id}", headers=_auth_headers()
    )
    assert delete_response.status_code == 200
    assert delete_response.json()["data"]["model_id"] == model_id

    missing_response = client.get(f"/api/v1/models/{model_id}", headers=_auth_headers())
    assert missing_response.status_code == 404
    assert missing_response.json()["error_code"] == "MODEL_NOT_FOUND"


def test_supporting_model_constraint_enforced(
    api_client: tuple[TestClient, ModelService],
) -> None:
    """Supporting models should reject supporting-model signal input dependencies."""

    client, _service = api_client

    payload = {
        "name": "Bad Supporting Model",
        "description": "invalid because of signal dependency",
        "model_type": "supporting_ml",
        "signal_type": "sentiment",
        "trainer_type": "sklearn",
        "algorithm": "random_forest",
        "hyperparameters": {"n_estimators": 100},
        "input_data_types": ["model_signal"],
        "input_frequency": "1m",
    }

    response = client.post("/api/v1/models", headers=_auth_headers(), json=payload)

    assert response.status_code == 400
    body = response.json()
    assert body["error_code"] == "MODEL_VALIDATION_ERROR"


def test_strategy_listing_and_details(
    api_client: tuple[TestClient, ModelService],
) -> None:
    """Strategy endpoints should return list and details for registered strategies."""

    client, service = api_client

    list_response = client.get("/api/v1/strategies", headers=_auth_headers())
    assert list_response.status_code == 200
    strategies = list_response.json()["data"]
    assert len(strategies) >= 1

    strategy_id = service.strategy_registry.get_all()[0].strategy_id
    detail_response = client.get(
        f"/api/v1/strategies/{strategy_id}",
        headers=_auth_headers(),
    )
    assert detail_response.status_code == 200
    detail = detail_response.json()["data"]
    assert detail["strategy_id"] == strategy_id
    assert detail["class_name"] == "_IntegrationStrategy"


def test_generation_listing_and_comparison(
    api_client: tuple[TestClient, ModelService],
) -> None:
    """Generation endpoints should list model history and compare selected runs."""

    client, service = api_client

    create_response = client.post(
        "/api/v1/models",
        headers=_auth_headers(),
        json=_core_model_payload(name="Generation Core Model"),
    )
    model_id = create_response.json()["data"]["model_id"]

    generation_one = service.generation_tracker.start_generation(
        model_id=model_id,
        hyperparameters={"learning_rate": 0.0003},
    )
    service.generation_tracker.complete_generation(
        generation_one.generation_id,
        training_metrics=TrainingMetrics(
            final_reward=1.1,
            mean_reward=0.8,
            std_reward=0.2,
            episodes_completed=10,
            timesteps_trained=5000,
            training_time_seconds=60.0,
        ),
        model_path="models/core/generation_1.zip",
    )
    service.generation_tracker.add_evaluation(
        generation_one.generation_id,
        EvaluationMetrics(sharpe_ratio=0.9, total_return=0.12, max_drawdown=0.08),
    )

    generation_two = service.generation_tracker.start_generation(
        model_id=model_id,
        hyperparameters={"learning_rate": 0.0002},
        parent_generation_id=generation_one.generation_id,
    )
    service.generation_tracker.complete_generation(
        generation_two.generation_id,
        training_metrics=TrainingMetrics(
            final_reward=1.4,
            mean_reward=1.0,
            std_reward=0.18,
            episodes_completed=11,
            timesteps_trained=5200,
            training_time_seconds=62.0,
        ),
        model_path="models/core/generation_2.zip",
    )
    service.generation_tracker.add_evaluation(
        generation_two.generation_id,
        EvaluationMetrics(sharpe_ratio=1.05, total_return=0.17, max_drawdown=0.06),
    )

    list_response = client.get(
        f"/api/v1/models/{model_id}/generations?page=1&page_size=20",
        headers=_auth_headers(),
    )
    assert list_response.status_code == 200
    listed = list_response.json()
    assert listed["total"] == 2
    assert len(listed["items"]) == 2

    compare_response = client.post(
        "/api/v1/generations/compare",
        headers=_auth_headers(),
        json={
            "model_id": model_id,
            "generation_ids": [
                generation_one.generation_id,
                generation_two.generation_id,
            ],
        },
    )
    assert compare_response.status_code == 200
    comparison = compare_response.json()["data"]
    assert len(comparison["generations"]) == 2
    assert "final_reward" in comparison["metric_deltas"]

    detail_response = client.get(
        f"/api/v1/generations/{generation_one.generation_id}",
        headers=_auth_headers(),
    )
    assert detail_response.status_code == 200
    detail = detail_response.json()["data"]
    assert detail["generation_id"] == generation_one.generation_id
    assert detail["model_id"] == model_id
