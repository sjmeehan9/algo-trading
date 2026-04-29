"""Integration tests for Phase 5.10 optimizer API endpoints."""

from __future__ import annotations

import json
from pathlib import Path

from algotrading.api.config import APIConfig
from algotrading.api.main import create_app
from algotrading.api.schemas.models import ModelConfigCreate, ModelType
from algotrading.api.services import ModelService
from algotrading.api.services.llm_optimizer import LLMOptimizer
from algotrading.src.models.registry import (
    CustomStrategyRegistry,
    SupportingModelRegistry,
)
from algotrading.src.models.tracking import (
    GenerationTracker,
    JsonFileStorage,
    TrainingMetrics,
)
from fastapi.testclient import TestClient


class _ApiOptimizerClient:
    """OpenAI-compatible test client returning deterministic JSON."""

    def complete_json(self, prompt: str) -> str:
        """Return one valid optimizer suggestion."""

        assert "Optimizer API Core" in prompt
        return json.dumps(
            {
                "summary": "Lower the learning rate before the next PPO generation.",
                "suggestions": [
                    {
                        "parameter": "learning_rate",
                        "current_value": 0.0003,
                        "suggested_value": 0.0001,
                        "rationale": "Reward improved while variance remained elevated.",
                        "confidence": "medium",
                        "expected_impact": "Smoother updates and lower drawdown.",
                    }
                ],
                "priority_changes": ["learning_rate"],
            }
        )


def _build_model_service(tmp_path: Path) -> ModelService:
    """Build an isolated model service for optimizer API tests."""

    return ModelService(
        supporting_registry=SupportingModelRegistry(),
        strategy_registry=CustomStrategyRegistry(strategy_dirs=[], auto_scan=False),
        generation_tracker=GenerationTracker(
            JsonFileStorage(str(tmp_path / "generations"))
        ),
        core_models_path=tmp_path / "core_models.json",
    )


def _auth_headers() -> dict[str, str]:
    """Return API authentication headers."""

    return {"X-API-Key": "optimizer-secret-key"}


def _create_model_and_generation(service: ModelService) -> str:
    """Create an optimizer-ready model and generation."""

    model = service.create_model(
        ModelConfigCreate(
            name="Optimizer API Core",
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
    generation = service.generation_tracker.start_generation(
        model_id=model.model_id,
        hyperparameters={"learning_rate": 0.0003, "n_steps": 2048},
    )
    service.generation_tracker.complete_generation(
        generation.generation_id,
        training_metrics=TrainingMetrics(
            final_reward=1.2,
            mean_reward=1.0,
            std_reward=0.3,
            episodes_completed=8,
            timesteps_trained=8000,
            training_time_seconds=90.0,
        ),
        model_path="models/api-optimizer/model.zip",
    )
    return model.model_id


def test_optimizer_api_analyze_latest_and_apply(tmp_path: Path) -> None:
    """Endpoint flow should analyze, retrieve latest suggestions, and apply one."""

    config = APIConfig(api_key="optimizer-secret-key", debug=True)
    app = create_app(config)
    model_service = _build_model_service(tmp_path)
    app.state.model_service = model_service
    app.state.optimizer_service = LLMOptimizer(
        model_service=model_service,
        openai_client=_ApiOptimizerClient(),
        suggestions_path=tmp_path / "optimizer_results.json",
    )
    model_id = _create_model_and_generation(model_service)

    with TestClient(app) as client:
        analyze_response = client.post(
            "/api/v1/optimizer/analyze",
            headers=_auth_headers(),
            json={"model_id": model_id, "max_generations": 3},
        )
        assert analyze_response.status_code == 201, analyze_response.text
        result = analyze_response.json()["data"]
        assert result["source"] == "openai"
        assert result["suggestions"][0]["parameter"] == "learning_rate"

        latest_response = client.get(
            f"/api/v1/optimizer/suggestions/{model_id}",
            headers=_auth_headers(),
        )
        assert latest_response.status_code == 200
        assert latest_response.json()["data"]["result_id"] == result["result_id"]

        apply_response = client.post(
            "/api/v1/optimizer/apply",
            headers=_auth_headers(),
            json={
                "model_id": model_id,
                "result_id": result["result_id"],
                "suggestion_id": result["suggestions"][0]["suggestion_id"],
            },
        )
        assert apply_response.status_code == 200, apply_response.text
        applied = apply_response.json()["data"]
        assert applied["suggestion"]["applied"] is True
        assert applied["updated_model"]["hyperparameters"]["learning_rate"] == 0.0001

        model_response = client.get(
            f"/api/v1/models/{model_id}", headers=_auth_headers()
        )
        assert model_response.status_code == 200
        assert (
            model_response.json()["data"]["hyperparameters"]["learning_rate"] == 0.0001
        )


def test_optimizer_api_missing_model_returns_404(tmp_path: Path) -> None:
    """Unknown model IDs should return standardized not-found errors."""

    app = create_app(APIConfig(api_key="optimizer-secret-key", debug=True))
    model_service = _build_model_service(tmp_path)
    app.state.model_service = model_service
    app.state.optimizer_service = LLMOptimizer(
        model_service=model_service,
        openai_client=_ApiOptimizerClient(),
        suggestions_path=tmp_path / "optimizer_results.json",
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/optimizer/analyze",
            headers=_auth_headers(),
            json={"model_id": "missing-model"},
        )

    assert response.status_code == 404
    assert response.json()["error_code"] == "MODEL_NOT_FOUND"
