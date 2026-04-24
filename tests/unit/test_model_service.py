"""Unit tests for model management service business rules."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from algotrading.api.schemas.models import ModelConfigCreate, ModelType
from algotrading.api.services import (
    InvalidModelStateError,
    ModelService,
    ModelValidationError,
)
from algotrading.src.data_pipeline import DataFrequency, DataType
from algotrading.src.models.registry import (
    CustomStrategyRegistry,
    ModelEntryConfig,
    ModelState,
    SupportingModelRegistry,
    trading_strategy,
)
from algotrading.src.models.signals import ModelSignal, SignalMetadata, SignalType
from algotrading.src.models.tracking import GenerationTracker, JsonFileStorage


@trading_strategy(name="Unit Strategy", signal_type=SignalType.POSITION)
class _UnitStrategy:
    """Simple strategy implementation for service-level tests."""

    def predict(self, state: dict[str, object], market_data: object) -> ModelSignal:
        del state, market_data
        return ModelSignal(
            timestamp=datetime.now(tz=UTC),
            signal_type=SignalType.POSITION,
            value=1.0,
            metadata=SignalMetadata(model_id="unit_strategy", model_type="strategy"),
        )


@pytest.fixture()
def service(tmp_path: Path) -> ModelService:
    """Create an isolated model service instance for each test."""

    supporting_registry = SupportingModelRegistry()
    strategy_registry = CustomStrategyRegistry(strategy_dirs=[], auto_scan=False)
    strategy_registry.register_strategy(_UnitStrategy)
    generation_tracker = GenerationTracker(
        JsonFileStorage(str(tmp_path / "generations"))
    )

    return ModelService(
        supporting_registry=supporting_registry,
        strategy_registry=strategy_registry,
        generation_tracker=generation_tracker,
        core_models_path=tmp_path / "core_models.json",
    )


def _core_payload() -> ModelConfigCreate:
    """Build a valid baseline core-RL payload."""

    return ModelConfigCreate(
        name="Core RL",
        description="core model",
        model_type=ModelType.CORE_RL,
        trainer_type="stable_baselines3",
        algorithm="ppo",
        hyperparameters={"learning_rate": 0.0003, "n_steps": 2048},
        training_data_config={"symbols": ["AAPL", "MSFT"]},
        supporting_model_ids=[],
        strategy_ids=[],
        environment_config={"initial_capital": 100000},
        reward_function="profit_seeker",
    )


def test_supporting_models_reject_signal_inputs(service: ModelService) -> None:
    """Supporting model validation should block signal dependency inputs."""

    payload = ModelConfigCreate(
        name="Bad Supporting",
        model_type=ModelType.SUPPORTING_ML,
        signal_type=SignalType.SENTIMENT,
        trainer_type="sklearn",
        algorithm="random_forest",
        hyperparameters={"n_estimators": 100},
        input_data_types=["model_signal"],
        input_frequency="1m",
    )

    with pytest.raises(ModelValidationError):
        service.create_model(payload)


def test_core_model_supporting_references_require_ready_state(
    service: ModelService,
) -> None:
    """Core models can reference supporting models only when they are READY."""

    supporting_model_id = "supporting-news-ready"
    service.supporting_registry.register(
        ModelEntryConfig(
            model_id=supporting_model_id,
            model_type="ml",
            signal_type=SignalType.SENTIMENT,
            trainer_class="tests.mocks.mock_ml_trainer.MockMLTrainer",
            input_data_types=[DataType.NEWS_TEXT],
            input_frequency=DataFrequency.MINUTE_1,
            description="news helper",
        )
    )

    strategy_id = service.strategy_registry.get_all()[0].strategy_id

    blocked_payload = _core_payload().model_copy(
        update={
            "supporting_model_ids": [supporting_model_id],
            "strategy_ids": [strategy_id],
        }
    )

    with pytest.raises(InvalidModelStateError):
        service.create_model(blocked_payload)

    service.supporting_registry.set_state(supporting_model_id, ModelState.LOADING)
    service.supporting_registry.set_state(supporting_model_id, ModelState.LOADED)
    service.supporting_registry.set_state(supporting_model_id, ModelState.READY)

    created = service.create_model(blocked_payload)

    assert created.supporting_model_ids == [supporting_model_id]
    assert created.strategy_ids == [strategy_id]


def test_list_models_applies_pagination(service: ModelService) -> None:
    """Model listing should return deterministic paginated slices."""

    for index in range(3):
        payload = _core_payload().model_copy(
            update={
                "name": f"Core RL {index}",
                "description": f"core model {index}",
            }
        )
        service.create_model(payload)

    page_one = service.list_models(page=1, page_size=2)
    page_two = service.list_models(page=2, page_size=2)

    assert page_one.total == 3
    assert page_one.pages == 2
    assert len(page_one.items) == 2
    assert len(page_two.items) == 1


def test_list_models_rejects_out_of_range_page(service: ModelService) -> None:
    """Pagination should reject page numbers beyond available range."""

    service.create_model(_core_payload())

    with pytest.raises(ModelValidationError):
        service.list_models(page=2, page_size=10)
