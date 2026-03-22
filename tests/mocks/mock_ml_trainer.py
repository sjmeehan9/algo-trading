"""Mock ML trainer for deterministic registry and inference tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from algotrading.src.trainers import (
    MLPrediction,
    MLTrainer,
    MLTrainingConfig,
    MLTrainingResult,
)


class MockMLTrainer(MLTrainer):
    """Simple in-memory ML trainer implementation for tests."""

    def __init__(self) -> None:
        self._trained = False

    def create_model(
        self,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        config: MLTrainingConfig,
    ) -> None:
        del input_shape, output_shape, config

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: MLTrainingConfig,
    ) -> MLTrainingResult:
        del X, y, config
        self._trained = True
        return MLTrainingResult(
            epochs_trained=1,
            final_loss=0.1,
            validation_loss=None,
            training_time_seconds=0.01,
        )

    def predict(self, X: np.ndarray) -> MLPrediction | list[MLPrediction]:
        self.ensure_trained()
        self.validate_input_shape(X)
        if X.ndim == 1:
            return MLPrediction(value=float(np.mean(X)), confidence=0.9)
        return [MLPrediction(value=float(np.mean(row)), confidence=0.9) for row in X]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.ensure_trained()
        self.validate_input_shape(X)
        samples = 1 if X.ndim == 1 else X.shape[0]
        return np.tile(np.array([0.4, 0.6], dtype=np.float64), (samples, 1))

    def save(self, filepath: str) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("mock-ml", encoding="utf-8")

    def load(self, filepath: str) -> None:
        del filepath
        self._trained = True

    @property
    def model_type(self) -> str:
        return "mock_ml"

    @property
    def is_trained(self) -> bool:
        return self._trained
