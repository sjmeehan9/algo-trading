"""Integration tests for custom strategy registry with unified model view."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from algotrading.src.data_pipeline import DataFrequency, DataType
from algotrading.src.models.registry import (
    CustomStrategyRegistry,
    ModelEntryConfig,
    SupportingModelRegistry,
)
from algotrading.src.models.signals import SignalType


def test_load_strategy_predict_and_unified_view(tmp_path: Path) -> None:
    """Strategies should load from file, emit signals, and appear in unified view."""

    strategies_dir = tmp_path / "strategies"
    strategies_dir.mkdir(parents=True)
    strategy_file = strategies_dir / "sample_strategy.py"

    strategy_file.write_text(
        "\n".join(
            [
                "from datetime import UTC, datetime",
                "from algotrading.src.models.registry import trading_strategy",
                "from algotrading.src.models.signals import ModelSignal, SignalMetadata, SignalType",
                "",
                "@trading_strategy(name='Sample Position Strategy', signal_type=SignalType.POSITION)",
                "class SamplePositionStrategy:",
                "    def predict(self, state, market_data) -> ModelSignal:",
                "        return ModelSignal(",
                "            timestamp=datetime.now(tz=UTC),",
                "            signal_type=SignalType.POSITION,",
                "            value=1.0,",
                "            confidence=0.9,",
                "            metadata=SignalMetadata(model_id='sample_position', model_type='strategy'),",
                "        )",
            ]
        ),
        encoding="utf-8",
    )

    strategy_registry = CustomStrategyRegistry(
        strategy_dirs=[str(strategies_dir)],
        auto_scan=True,
    )

    strategy_entries = strategy_registry.get_all()
    assert len(strategy_entries) == 1

    strategy_entry = strategy_entries[0]
    strategy_instance = strategy_registry.get_instance(strategy_entry.strategy_id)
    signal = strategy_instance.predict(
        state={"symbol": "AAPL"},
        market_data=pd.DataFrame({"close": [100.0, 101.0]}),
    )

    assert signal.signal_type == SignalType.POSITION
    assert signal.value == 1.0

    model_registry = SupportingModelRegistry()
    model_registry.register(
        ModelEntryConfig(
            model_id="ml-news",
            model_type="ml",
            signal_type=SignalType.SENTIMENT,
            trainer_class="tests.mocks.mock_ml_trainer.MockMLTrainer",
            input_data_types=[DataType.NEWS_TEXT],
            input_frequency=DataFrequency.MINUTE_1,
            description="ML news model",
        )
    )

    unified = strategy_registry.get_unified_view(model_registry=model_registry)

    assert len(unified) == 2
    assert any(entry["model_type"] == "ml" for entry in unified)
    assert any(entry["model_type"] == "strategy" for entry in unified)
