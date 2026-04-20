"""Unit tests for supporting model registry behavior."""

from __future__ import annotations

from pathlib import Path

import pytest
from algotrading.src.data_pipeline import DataFrequency, DataType
from algotrading.src.models.registry import (
    InvalidDependencyError,
    ModelAlreadyExistsError,
    ModelEntryConfig,
    ModelState,
    ModelStateError,
    SupportingModelRegistry,
)
from algotrading.src.models.signals import SignalType


def _build_config(
    model_id: str,
    model_type: str = "ml",
    signal_type: SignalType = SignalType.SENTIMENT,
    trainer_class: str = "tests.mocks.mock_ml_trainer.MockMLTrainer",
    input_data_types: list[DataType] | None = None,
) -> ModelEntryConfig:
    return ModelEntryConfig(
        model_id=model_id,
        model_type=model_type,
        signal_type=signal_type,
        trainer_class=trainer_class,
        input_data_types=input_data_types or [DataType.NEWS_TEXT],
        input_frequency=DataFrequency.MINUTE_1,
        description="test model",
        config={"param": 1},
    )


def test_register_model_with_valid_config() -> None:
    """Registry accepts valid model configurations."""

    registry = SupportingModelRegistry()
    model_id = registry.register(_build_config(model_id="news-sentiment-ml"))

    assert model_id == "news-sentiment-ml"
    entry = registry.get(model_id)
    assert entry is not None
    assert entry.config.model_type == "ml"
    assert entry.state == ModelState.REGISTERED


def test_duplicate_model_id_rejected() -> None:
    """Duplicate model IDs raise ModelAlreadyExistsError."""

    registry = SupportingModelRegistry()
    config = _build_config(model_id="dup-model")
    registry.register(config)

    with pytest.raises(ModelAlreadyExistsError):
        registry.register(config)


def test_invalid_supporting_dependency_detection() -> None:
    """Supporting model signal dependencies are rejected."""

    registry = SupportingModelRegistry()
    invalid = _build_config(
        model_id="bad-dependency",
        input_data_types=[DataType.MARKET_BAR, DataType.SIGNAL],
    )

    with pytest.raises(InvalidDependencyError):
        registry.register(invalid)


def test_discovery_methods_filter_entries() -> None:
    """Discovery APIs return expected model subsets."""

    registry = SupportingModelRegistry()
    registry.register(
        _build_config(
            model_id="ml-sentiment",
            model_type="ml",
            signal_type=SignalType.SENTIMENT,
            trainer_class="tests.mocks.mock_ml_trainer.MockMLTrainer",
            input_data_types=[DataType.NEWS_TEXT],
        )
    )
    registry.register(
        _build_config(
            model_id="rl-trend",
            model_type="rl",
            signal_type=SignalType.TREND,
            trainer_class="tests.mocks.mock_rl_trainer.MockRLTrainer",
            input_data_types=[DataType.MARKET_BAR],
        )
    )

    assert len(registry.get_all()) == 2
    assert [entry.config.model_id for entry in registry.find_by_type("ml")] == [
        "ml-sentiment"
    ]
    assert [
        entry.config.model_id
        for entry in registry.find_by_signal_type(SignalType.TREND)
    ] == ["rl-trend"]

    registry.set_state("rl-trend", ModelState.LOADING)
    registry.set_state("rl-trend", ModelState.LOADED)
    registry.set_state("rl-trend", ModelState.READY)
    assert [entry.config.model_id for entry in registry.find_ready()] == ["rl-trend"]


def test_state_transition_validation_enforced() -> None:
    """Invalid model state transitions raise ModelStateError."""

    registry = SupportingModelRegistry()
    registry.register(_build_config(model_id="stateful"))

    registry.set_state("stateful", ModelState.LOADING)
    registry.set_state("stateful", ModelState.LOADED)

    with pytest.raises(ModelStateError):
        registry.set_state("stateful", ModelState.REGISTERED)


def test_load_unload_model_sets_lifecycle_fields() -> None:
    """Loading attaches trainer instance and unloading clears runtime data."""

    registry = SupportingModelRegistry()
    registry.register(_build_config(model_id="loadable"))

    registry.load_model("loadable")
    entry = registry.get("loadable")
    assert entry is not None
    assert entry.state == ModelState.LOADED
    assert entry.trainer is not None
    assert entry.loaded_at is not None

    registry.unload_model("loadable")
    entry = registry.get("loadable")
    assert entry is not None
    assert entry.state == ModelState.UNLOADED
    assert entry.trainer is None
    assert entry.loaded_at is None


def test_registry_persistence_round_trip(tmp_path: Path) -> None:
    """Registry save/load persists model configuration and lifecycle metadata."""

    config_path = tmp_path / "supporting_registry.json"
    original = SupportingModelRegistry(config_path=str(config_path))
    original.register(_build_config(model_id="persisted"))
    original.set_state("persisted", ModelState.LOADING)
    original.set_state("persisted", ModelState.ERROR, error="load failure")

    loaded = SupportingModelRegistry()
    loaded.load_config(str(config_path))

    entry = loaded.get("persisted")
    assert entry is not None
    assert entry.config.model_id == "persisted"
    assert entry.state == ModelState.ERROR
    assert entry.error_message == "load failure"


def test_state_change_callbacks_include_registration_and_updates() -> None:
    """State callbacks fire for registration and subsequent state changes."""

    events: list[tuple[str, ModelState]] = []
    registry = SupportingModelRegistry()
    registry.on_state_change(lambda model_id, state: events.append((model_id, state)))

    registry.register(_build_config(model_id="callbacks"))
    registry.set_state("callbacks", ModelState.LOADING)

    assert events[0] == ("callbacks", ModelState.REGISTERED)
    assert events[1] == ("callbacks", ModelState.LOADING)
