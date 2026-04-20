"""Integration tests for end-to-end supporting model registry workflows."""

from __future__ import annotations

from pathlib import Path

from algotrading.src.data_pipeline import DataFrequency, DataType
from algotrading.src.models.registry import (
    ModelEntryConfig,
    ModelState,
    SupportingModelRegistry,
)
from algotrading.src.models.signals import SignalType


def test_register_load_and_query_models(tmp_path: Path) -> None:
    """Registry supports register, load, ready query, and persistence flow."""

    config_path = tmp_path / "registry.json"
    registry = SupportingModelRegistry(config_path=str(config_path))

    ml_config = ModelEntryConfig(
        model_id="news-support-model",
        model_type="ml",
        signal_type=SignalType.SENTIMENT,
        trainer_class="tests.mocks.mock_ml_trainer.MockMLTrainer",
        model_path=None,
        config={"window": 8},
        input_data_types=[DataType.NEWS_TEXT],
        input_frequency=DataFrequency.MINUTE_1,
        description="news sentiment supporting model",
    )
    rl_config = ModelEntryConfig(
        model_id="trend-support-model",
        model_type="rl",
        signal_type=SignalType.TREND,
        trainer_class="tests.mocks.mock_rl_trainer.MockRLTrainer",
        model_path=None,
        config={"lookback": 32},
        input_data_types=[DataType.MARKET_BAR, DataType.INDICATOR],
        input_frequency=DataFrequency.SECOND_5,
        description="trend support model",
    )

    registry.register(ml_config)
    registry.register(rl_config)

    registry.load_model("news-support-model")
    registry.set_state("news-support-model", ModelState.READY)

    ready_models = registry.find_ready()
    assert [entry.config.model_id for entry in ready_models] == ["news-support-model"]

    assert len(registry.find_by_type("ml")) == 1
    assert len(registry.find_by_signal_type(SignalType.TREND)) == 1

    restored = SupportingModelRegistry()
    restored.load_config(str(config_path))

    restored_news = restored.get("news-support-model")
    restored_trend = restored.get("trend-support-model")
    assert restored_news is not None
    assert restored_trend is not None
    assert restored_news.state == ModelState.READY
    assert restored_news.trainer is None
    assert restored_trend.state == ModelState.REGISTERED
