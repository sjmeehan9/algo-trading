"""Unit tests for custom strategy decorator metadata behavior."""

from __future__ import annotations

from datetime import UTC, datetime

from algotrading.src.models.registry import trading_strategy
from algotrading.src.models.signals import ModelSignal, SignalMetadata, SignalType


def test_trading_strategy_decorator_attaches_metadata() -> None:
    """Decorator should attach expected strategy metadata."""

    @trading_strategy(
        name="Decorated Strategy",
        signal_type=SignalType.POSITION,
        description="Test description",
        version="2.0.0",
    )
    class DecoratedStrategy:
        def predict(self, state: dict[str, object], market_data: object) -> ModelSignal:
            return ModelSignal(
                timestamp=datetime.now(tz=UTC),
                signal_type=SignalType.POSITION,
                value=0.0,
                metadata=SignalMetadata(model_id="decorated", model_type="strategy"),
            )

    metadata = getattr(DecoratedStrategy, "_strategy_metadata")
    assert metadata["is_trading_strategy"] is True
    assert metadata["name"] == "Decorated Strategy"
    assert metadata["signal_type"] == SignalType.POSITION
    assert metadata["description"] == "Test description"
    assert metadata["version"] == "2.0.0"


def test_trading_strategy_decorator_defaults() -> None:
    """Decorator should derive default name and description."""

    @trading_strategy()
    class Defaulted:
        """Default strategy docstring."""

        def predict(self, state: dict[str, object], market_data: object) -> ModelSignal:
            return ModelSignal(
                timestamp=datetime.now(tz=UTC),
                signal_type=SignalType.POSITION,
                value=0.0,
                metadata=SignalMetadata(model_id="defaulted", model_type="strategy"),
            )

    metadata = getattr(Defaulted, "_strategy_metadata")
    assert metadata["name"] == "Defaulted"
    assert metadata["signal_type"] == SignalType.POSITION
    assert metadata["description"] == "Default strategy docstring."
    assert metadata["version"] == "1.0.0"
