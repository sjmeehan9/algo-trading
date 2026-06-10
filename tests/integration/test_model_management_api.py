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

    supporting_registry = SupportingModelRegistry(
        config_path=str(tmp_path / "supporting_models_registry.json")
    )
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


def _supporting_sentiment_payload(
    name: str = "News Sentiment Model",
    algorithm: str = "vader",
) -> dict[str, object]:
    """Return a valid supporting sentiment model API payload."""

    return {
        "name": name,
        "description": "news sentiment supporting model",
        "model_type": "supporting_ml",
        "signal_type": "sentiment",
        "trainer_type": "huggingface",
        "algorithm": algorithm,
        "hyperparameters": {},
        "input_data_types": ["news_text"],
        "input_frequency": "1m",
    }


def _create_supporting_model(
    client: TestClient,
    payload: dict[str, object] | None = None,
) -> str:
    """Create a supporting sentiment model and return its ID."""

    response = client.post(
        "/api/v1/models",
        headers=_auth_headers(),
        json=payload or _supporting_sentiment_payload(),
    )
    assert response.status_code == 201, response.text
    return response.json()["data"]["model_id"]


def test_supporting_lifecycle_activate_and_unload(
    api_client: tuple[TestClient, ModelService],
) -> None:
    """Activate a pretrained sentiment model, verify READY, unload, verify state."""

    client, service = api_client

    model_id = _create_supporting_model(client)

    initial = client.get(
        f"/api/v1/models/{model_id}/lifecycle", headers=_auth_headers()
    )
    assert initial.status_code == 200
    assert initial.json()["data"]["state"] == "registered"
    assert initial.json()["data"]["is_ready"] is False

    activate = client.post(
        f"/api/v1/models/{model_id}/activate-pretrained",
        headers=_auth_headers(),
        json={"hyperparameters": {}},
    )
    assert activate.status_code == 200, activate.text
    activated = activate.json()["data"]
    assert activated["state"] == "ready"
    assert activated["is_ready"] is True
    assert activated["signal_type"] == "sentiment"
    assert any(
        check["name"] == "trainer_loaded" and check["passed"]
        for check in activated["readiness_checks"]
    )

    # State change must persist to the registry JSON.
    assert service.get_supporting_entry(model_id).state.value == "ready"

    unload = client.post(f"/api/v1/models/{model_id}/unload", headers=_auth_headers())
    assert unload.status_code == 200, unload.text
    assert unload.json()["data"]["state"] == "unloaded"
    assert unload.json()["data"]["is_ready"] is False

    lifecycle = client.get(
        f"/api/v1/models/{model_id}/lifecycle", headers=_auth_headers()
    )
    assert lifecycle.json()["data"]["state"] == "unloaded"
    assert service.get_supporting_entry(model_id).state.value == "unloaded"


def test_supporting_lifecycle_load_artifact(
    api_client: tuple[TestClient, ModelService],
    tmp_path: Path,
) -> None:
    """Load a saved NewsSentimentTrainer artifact and reach READY state."""

    from algotrading.src.models.supporting.sentiment.config import (
        SentimentConfig,
        SentimentModelType,
    )
    from algotrading.src.models.supporting.sentiment.news_sentiment import (
        NewsSentimentTrainer,
    )

    client, service = api_client

    artifact_dir = tmp_path / "sentiment_artifact"
    trainer = NewsSentimentTrainer(
        config=SentimentConfig(model_type=SentimentModelType.VADER)
    )
    trainer.save(str(artifact_dir))
    assert (artifact_dir / "news_sentiment.json").exists()

    model_id = _create_supporting_model(client)

    response = client.post(
        f"/api/v1/models/{model_id}/load-artifact",
        headers=_auth_headers(),
        json={"model_path": str(artifact_dir)},
    )
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["state"] == "ready"
    assert data["is_ready"] is True
    assert data["model_path"] == str(artifact_dir)

    entry = service.get_supporting_entry(model_id)
    assert entry.state.value == "ready"
    assert entry.config.model_path == str(artifact_dir)


def test_supporting_lifecycle_missing_artifact_returns_400(
    api_client: tuple[TestClient, ModelService],
    tmp_path: Path,
) -> None:
    """A non-existent artifact path should fail validation with 400."""

    client, _service = api_client
    model_id = _create_supporting_model(client)

    response = client.post(
        f"/api/v1/models/{model_id}/load-artifact",
        headers=_auth_headers(),
        json={"model_path": str(tmp_path / "does_not_exist")},
    )
    assert response.status_code == 400
    assert response.json()["error_code"] == "LIFECYCLE_VALIDATION_ERROR"


def test_supporting_lifecycle_rejects_unsupported_algorithm(
    api_client: tuple[TestClient, ModelService],
) -> None:
    """Activation must reject algorithms without a real pretrained backend."""

    client, _service = api_client

    payload = {
        "name": "RF Sentiment",
        "description": "unsupported pretrained backend",
        "model_type": "supporting_ml",
        "signal_type": "sentiment",
        "trainer_type": "sklearn",
        "algorithm": "random_forest",
        "hyperparameters": {"n_estimators": 50},
        "input_data_types": ["news_text"],
        "input_frequency": "1m",
    }
    model_id = _create_supporting_model(client, payload)

    response = client.post(
        f"/api/v1/models/{model_id}/activate-pretrained",
        headers=_auth_headers(),
        json={"hyperparameters": {}},
    )
    assert response.status_code == 409
    assert response.json()["error_code"] == "MODEL_TYPE_UNSUPPORTED"


def _finbert_available() -> bool:
    """Return True when the transformers dependency is importable."""

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.requires_finbert
@pytest.mark.skipif(
    not _finbert_available(),
    reason="transformers/torch are required for the FinBERT activation path.",
)
def test_supporting_lifecycle_activate_finbert(
    api_client: tuple[TestClient, ModelService],
) -> None:
    """Activate a FinBERT sentiment model end to end through the API.

    This exercises the real transformer-sentiment path: SentimentConfig is
    built from hyperparameters, the FinBERT model is downloaded/loaded, the
    smoke prediction runs, and the registry entry transitions to READY with
    state persisted. The test skips (rather than fails) when the model cannot
    be downloaded, so it is safe in offline environments.
    """

    client, service = api_client

    payload = {
        "name": "FinBERT Sentiment",
        "description": "transformer sentiment supporting model",
        "model_type": "supporting_ml",
        "signal_type": "sentiment",
        "trainer_type": "huggingface",
        "algorithm": "transformer_sentiment",
        "hyperparameters": {
            "model_name": "ProsusAI/finbert",
            "max_length": 256,
            "device": "cpu",
        },
        "input_data_types": ["news_text"],
        "input_frequency": "1m",
    }
    model_id = _create_supporting_model(client, payload)

    response = client.post(
        f"/api/v1/models/{model_id}/activate-pretrained",
        headers=_auth_headers(),
        json={},
    )

    if response.status_code == 409:
        body = response.json()
        if body.get("error_code") == "LIFECYCLE_OPERATION_ERROR":
            pytest.skip(
                "FinBERT model could not be downloaded/loaded in this "
                f"environment: {body.get('error')}"
            )

    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["state"] == "ready"
    assert data["is_ready"] is True
    assert data["algorithm"] == "transformer_sentiment"

    entry = service.get_supporting_entry(model_id)
    assert entry.state.value == "ready"
    assert entry.config.config.get("sentiment_backend") == "finbert"

    # The loaded backend must return a real, correctly-signed sentiment result.
    prediction = entry.trainer.predict(
        "The company beat earnings expectations and raised guidance."
    )
    value = (
        prediction.value if not isinstance(prediction, list) else prediction[0].value
    )
    assert value > 0.0


def test_supporting_lifecycle_rejects_core_model(
    api_client: tuple[TestClient, ModelService],
) -> None:
    """Lifecycle endpoints must reject core RL models with a 409 error."""

    client, _service = api_client

    create_response = client.post(
        "/api/v1/models",
        headers=_auth_headers(),
        json=_core_model_payload(name="Core For Lifecycle"),
    )
    model_id = create_response.json()["data"]["model_id"]

    lifecycle = client.get(
        f"/api/v1/models/{model_id}/lifecycle", headers=_auth_headers()
    )
    assert lifecycle.status_code == 409
    assert lifecycle.json()["error_code"] == "MODEL_TYPE_UNSUPPORTED"

    activate = client.post(
        f"/api/v1/models/{model_id}/activate-pretrained",
        headers=_auth_headers(),
        json={},
    )
    assert activate.status_code == 409


def test_supporting_lifecycle_unknown_model_returns_404(
    api_client: tuple[TestClient, ModelService],
) -> None:
    """Lifecycle endpoints must return 404 for unknown model IDs."""

    client, _service = api_client

    response = client.get(
        "/api/v1/models/supporting-ml-missing/lifecycle", headers=_auth_headers()
    )
    assert response.status_code == 404
    assert response.json()["error_code"] == "MODEL_NOT_FOUND"


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
