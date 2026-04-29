"""Unit tests for LLM hyperparameter optimizer parsing and apply flow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from algotrading.api.schemas.models import ModelConfigCreate, ModelType
from algotrading.api.schemas.optimizer import (
    ApplySuggestionRequest,
    OptimizerAnalyzeRequest,
    SuggestionOutcomeStatus,
)
from algotrading.api.services import ModelService
from algotrading.api.services.llm_optimizer import LLMOptimizer
from algotrading.src.models.registry import (
    CustomStrategyRegistry,
    SupportingModelRegistry,
)
from algotrading.src.models.tracking import (
    EvaluationMetrics,
    GenerationTracker,
    JsonFileStorage,
    TrainingMetrics,
)


class _MockOpenAIClient:
    """OpenAI-compatible test double returning a configured response."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def complete_json(self, prompt: str) -> str:
        """Capture prompt text and return the configured completion."""

        self.prompts.append(prompt)
        return self.response


@pytest.fixture()
def model_service(tmp_path: Path) -> ModelService:
    """Create an isolated model service for optimizer unit tests."""

    return ModelService(
        supporting_registry=SupportingModelRegistry(),
        strategy_registry=CustomStrategyRegistry(strategy_dirs=[], auto_scan=False),
        generation_tracker=GenerationTracker(
            JsonFileStorage(str(tmp_path / "generations"))
        ),
        core_models_path=tmp_path / "core_models.json",
    )


def _create_model(service: ModelService) -> str:
    """Create a baseline core RL model and return its model ID."""

    model = service.create_model(
        ModelConfigCreate(
            name="Optimizer Core",
            model_type=ModelType.CORE_RL,
            trainer_type="stable_baselines3",
            algorithm="ppo",
            hyperparameters={"learning_rate": 0.0003, "gamma": 0.98, "n_steps": 2048},
            training_data_config={"symbols": ["AAPL"]},
            supporting_model_ids=[],
            strategy_ids=[],
            environment_config={"initial_capital": 100000},
            reward_function="profit_seeker",
        )
    )
    return model.model_id


def _add_generation(
    service: ModelService,
    model_id: str,
    *,
    learning_rate: float,
    final_reward: float,
    sharpe_ratio: float,
) -> str:
    """Create a completed and evaluated generation for a model."""

    generation = service.generation_tracker.start_generation(
        model_id=model_id,
        hyperparameters={
            "learning_rate": learning_rate,
            "gamma": 0.98,
            "n_steps": 2048,
        },
    )
    service.generation_tracker.complete_generation(
        generation.generation_id,
        training_metrics=TrainingMetrics(
            final_reward=final_reward,
            mean_reward=final_reward - 0.2,
            std_reward=0.1,
            episodes_completed=5,
            timesteps_trained=5000,
            training_time_seconds=45.0,
        ),
        model_path=f"models/{model_id}/{generation.generation_id}.zip",
    )
    service.generation_tracker.add_evaluation(
        generation.generation_id,
        EvaluationMetrics(
            sharpe_ratio=sharpe_ratio,
            total_return=0.08 + sharpe_ratio / 100,
            max_drawdown=0.05,
        ),
    )
    return generation.generation_id


def _optimizer_response() -> str:
    """Return a valid JSON optimizer response for tests."""

    return json.dumps(
        {
            "summary": "Reward improved, but a smaller learning rate may reduce drawdown.",
            "suggestions": [
                {
                    "parameter": "learning_rate",
                    "current_value": 0.0003,
                    "suggested_value": 0.00015,
                    "rationale": "Latest generation improved but remained noisy.",
                    "confidence": "high",
                    "expected_impact": "Lower variance in policy updates.",
                }
            ],
            "priority_changes": ["learning_rate"],
        }
    )


def test_build_prompt_includes_model_history_and_schema(
    model_service: ModelService,
) -> None:
    """Prompt construction should include model, generations, trends, and schema."""

    model_id = _create_model(model_service)
    _add_generation(
        model_service,
        model_id,
        learning_rate=0.0003,
        final_reward=1.0,
        sharpe_ratio=0.8,
    )
    _add_generation(
        model_service,
        model_id,
        learning_rate=0.0002,
        final_reward=1.5,
        sharpe_ratio=1.1,
    )
    optimizer = LLMOptimizer(model_service=model_service)
    model = model_service.get_model(model_id)
    generations = model_service.generation_tracker.get_generations_for_model(model_id)

    prompt = optimizer.build_prompt(
        model=model,
        generations=generations,
        include_backtest_metrics=True,
    )
    payload = json.loads(prompt)

    assert payload["model"]["model_id"] == model_id
    assert len(payload["history"]) == 2
    assert "response_schema" in payload
    assert payload["metric_trends"]["final_reward_delta"] == pytest.approx(0.5)


def test_parse_response_supports_json_markdown_and_plain_text(
    model_service: ModelService,
) -> None:
    """Response parsing should handle valid JSON, fenced JSON, and plain text."""

    model_id = _create_model(model_service)
    model = model_service.get_model(model_id)
    optimizer = LLMOptimizer(model_service=model_service)

    parsed = optimizer.parse_response(_optimizer_response(), model)
    assert parsed.suggestions[0].parameter == "learning_rate"
    assert parsed.suggestions[0].confidence.value == "high"

    fenced = optimizer.parse_response(f"```json\n{_optimizer_response()}\n```", model)
    assert fenced.priority_changes == ["learning_rate"]

    plain = optimizer.parse_response("Try smaller updates and compare drawdown.", model)
    assert plain.suggestions == []
    assert plain.summary == "Try smaller updates and compare drawdown."


def test_analyze_persists_openai_suggestions(
    model_service: ModelService, tmp_path: Path
) -> None:
    """Analyzer should call the injected client, parse suggestions, and persist them."""

    model_id = _create_model(model_service)
    _add_generation(
        model_service,
        model_id,
        learning_rate=0.0003,
        final_reward=1.0,
        sharpe_ratio=0.8,
    )
    client = _MockOpenAIClient(_optimizer_response())
    suggestions_path = tmp_path / "optimizer.json"
    optimizer = LLMOptimizer(
        model_service=model_service,
        openai_client=client,
        suggestions_path=suggestions_path,
    )

    result = optimizer.analyze(OptimizerAnalyzeRequest(model_id=model_id))

    assert result.source.value == "openai"
    assert result.suggestions[0].suggested_value == pytest.approx(0.00015)
    assert len(client.prompts) == 1

    reloaded = LLMOptimizer(
        model_service=model_service, suggestions_path=suggestions_path
    )
    latest = reloaded.get_latest_result(model_id)
    assert latest is not None
    assert latest.result_id == result.result_id


def test_apply_suggestion_updates_model_and_marks_pending(
    model_service: ModelService,
    tmp_path: Path,
) -> None:
    """Applying a suggestion should update hyperparameters and track status."""

    model_id = _create_model(model_service)
    _add_generation(
        model_service,
        model_id,
        learning_rate=0.0003,
        final_reward=1.0,
        sharpe_ratio=0.8,
    )
    optimizer = LLMOptimizer(
        model_service=model_service,
        openai_client=_MockOpenAIClient(_optimizer_response()),
        suggestions_path=tmp_path / "optimizer.json",
    )
    result = optimizer.analyze(OptimizerAnalyzeRequest(model_id=model_id))
    suggestion_id = result.suggestions[0].suggestion_id

    response = optimizer.apply_suggestion(
        ApplySuggestionRequest(
            model_id=model_id,
            result_id=result.result_id,
            suggestion_id=suggestion_id,
        )
    )

    assert response.updated_model.hyperparameters["learning_rate"] == pytest.approx(
        0.00015
    )
    assert response.suggestion.applied is True
    assert (
        response.suggestion.outcome_status
        == SuggestionOutcomeStatus.PENDING_NEXT_GENERATION
    )
