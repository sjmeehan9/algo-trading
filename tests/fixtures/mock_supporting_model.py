"""Deterministic supporting-model fixtures for integration tests."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from algotrading.src.models.registry import trading_strategy
from algotrading.src.models.signals import ModelSignal, SignalMetadata, SignalType
from algotrading.src.trainers import (
    MLPrediction,
    MLTrainer,
    MLTrainingConfig,
    MLTrainingResult,
)


class MockSupportingModel(MLTrainer):
    """Small ML trainer used for deterministic supporting-model integration tests."""

    def __init__(
        self,
        signal_value: float = 0.45,
        confidence: float = 0.82,
        delay_seconds: float = 0.0,
    ) -> None:
        """Initialize predictable mock inference behavior."""

        self._signal_value = float(signal_value)
        self._confidence = float(np.clip(confidence, 0.0, 1.0))
        self._delay_seconds = float(max(0.0, delay_seconds))
        self._trained = True
        self.predict_count = 0

    def create_model(
        self,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        config: MLTrainingConfig,
    ) -> None:
        """No-op model creation for interface compatibility."""

        del input_shape, output_shape, config

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: MLTrainingConfig,
    ) -> MLTrainingResult:
        """Mark trainer as fitted and return deterministic training metadata."""

        del X, y, config
        self._trained = True
        return MLTrainingResult(
            epochs_trained=1,
            final_loss=0.0,
            validation_loss=0.0,
            training_time_seconds=0.0,
        )

    def predict(self, X: np.ndarray | object) -> MLPrediction | list[MLPrediction]:
        """Return deterministic prediction values from array or dict payloads."""

        self.ensure_trained()
        self.predict_count += 1

        if self._delay_seconds > 0:
            time.sleep(self._delay_seconds)

        if isinstance(X, np.ndarray):
            self.validate_input_shape(X)
            if X.ndim == 1:
                return MLPrediction(
                    value=self._signal_value, confidence=self._confidence
                )
            return [
                MLPrediction(value=self._signal_value, confidence=self._confidence)
                for _ in X
            ]

        value = self._signal_value
        if isinstance(X, dict):
            text = f"{X.get('headline', '')} {X.get('body', '')}".lower()
            if "strong" in text or "beat" in text or "raises" in text:
                value = max(value, 0.7)
            if "weak" in text or "miss" in text or "cuts" in text:
                value = -abs(value)

        return MLPrediction(
            value=float(np.clip(value, -1.0, 1.0)), confidence=self._confidence
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return fixed binary probabilities for compatibility with callers."""

        self.ensure_trained()
        self.validate_input_shape(X)
        samples = 1 if X.ndim == 1 else X.shape[0]
        return np.tile(np.array([0.4, 0.6], dtype=np.float64), (samples, 1))

    def save(self, filepath: str) -> None:
        """Persist lightweight marker artifact for save/load tests."""

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("mock-supporting-model", encoding="utf-8")

    def load(self, filepath: str) -> None:
        """Load model marker and restore trained state."""

        del filepath
        self._trained = True

    @property
    def model_type(self) -> str:
        """Return trainer type identifier."""

        return "mock_supporting_ml"

    @property
    def is_trained(self) -> bool:
        """Return whether mock model is available for predictions."""

        return self._trained


@trading_strategy(
    name="Mock Supporting Strategy",
    signal_type=SignalType.POSITION,
    description="Deterministic long-position strategy for integration testing",
)
class MockSupportingStrategy:
    """Strategy fixture returning a deterministic POSITION signal."""

    def predict(self, state: dict[str, Any], market_data: object) -> ModelSignal:
        """Build a deterministic long-position signal from incoming payload data."""

        _ = state
        timestamp = getattr(market_data, "timestamp", datetime.now(tz=UTC))
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            timestamp = timestamp.replace(tzinfo=UTC)

        symbol = getattr(market_data, "symbol", None)
        return ModelSignal(
            timestamp=timestamp,
            signal_type=SignalType.POSITION,
            value=1.0,
            confidence=1.0,
            symbol=symbol,
            metadata=SignalMetadata(
                model_id="mock_supporting_strategy",
                model_type="strategy",
            ),
        )


__all__ = ["MockSupportingModel", "MockSupportingStrategy"]
