"""Unit tests for ML trainer interface abstractions."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest
from algotrading.src.trainers import (
    MLPrediction,
    MLTrainer,
    MLTrainingConfig,
    MLTrainingResult,
    ModelNotTrainedError,
)


class _BaseSuperHarness(MLTrainer):
    """Concrete harness delegating each abstract contract to super."""

    def create_model(
        self,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        config: MLTrainingConfig,
    ) -> None:
        super().create_model(input_shape, output_shape, config)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: MLTrainingConfig,
    ) -> MLTrainingResult:
        return super().train(X, y, config)

    def predict(self, X: np.ndarray) -> MLPrediction | list[MLPrediction]:
        return super().predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return super().predict_proba(X)

    def save(self, filepath: str) -> None:
        super().save(filepath)

    def load(self, filepath: str) -> None:
        super().load(filepath)

    @property
    def model_type(self) -> str:
        return super().model_type

    @property
    def is_trained(self) -> bool:
        return super().is_trained


class _ConcreteMLTrainer(MLTrainer):
    """Simple concrete trainer used to exercise helper methods."""

    def __init__(self, *, trained: bool) -> None:
        self._trained = trained

    def create_model(
        self,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        config: MLTrainingConfig,
    ) -> None:
        _ = input_shape
        _ = output_shape
        _ = config

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: MLTrainingConfig,
    ) -> MLTrainingResult:
        self.validate_input_shape(X)
        self.validate_input_shape(y)
        self._trained = True
        _ = config
        return MLTrainingResult(
            epochs_trained=1,
            final_loss=0.1,
            validation_loss=0.12,
            training_time_seconds=0.01,
            model_path=None,
            metrics={"accuracy": 0.9},
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
        return np.tile(np.array([0.25, 0.75], dtype=np.float64), (samples, 1))

    def save(self, filepath: str) -> None:
        _ = filepath

    def load(self, filepath: str) -> None:
        _ = filepath
        self._trained = True

    @property
    def model_type(self) -> str:
        return "mock_ml"

    @property
    def is_trained(self) -> bool:
        return self._trained


def test_ml_trainer_abc_cannot_be_instantiated_directly() -> None:
    """MLTrainer is abstract and cannot be instantiated directly."""

    with pytest.raises(TypeError):
        MLTrainer()


def test_abstract_methods_raise_not_implemented_when_delegating_to_super() -> None:
    """Abstract members raise NotImplementedError via super delegation."""

    trainer = _BaseSuperHarness()
    config = MLTrainingConfig(epochs=1)
    X = np.array([[1.0, 2.0]], dtype=np.float64)
    y = np.array([1.0], dtype=np.float64)

    with pytest.raises(NotImplementedError):
        trainer.create_model((2,), (1,), config)
    with pytest.raises(NotImplementedError):
        trainer.train(X, y, config)
    with pytest.raises(NotImplementedError):
        trainer.predict(X)
    with pytest.raises(NotImplementedError):
        trainer.predict_proba(X)
    with pytest.raises(NotImplementedError):
        trainer.save("model.pkl")
    with pytest.raises(NotImplementedError):
        trainer.load("model.pkl")
    with pytest.raises(NotImplementedError):
        _ = trainer.model_type
    with pytest.raises(NotImplementedError):
        _ = trainer.is_trained


def test_ml_dataclasses_store_expected_values() -> None:
    """ML trainer dataclasses store and expose expected fields."""

    config = MLTrainingConfig(
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        early_stopping=True,
        custom_params={"dropout": 0.1},
    )
    result = MLTrainingResult(
        epochs_trained=8,
        final_loss=0.3,
        validation_loss=0.35,
        training_time_seconds=3.4,
        model_path="/tmp/model.pkl",
        metrics={"accuracy": 0.88, "f1": 0.82},
    )
    prediction = MLPrediction(
        value="BUY",
        confidence=0.77,
        probabilities=np.array([0.23, 0.77], dtype=np.float64),
    )

    assert asdict(config)["epochs"] == 10
    assert config.custom_params == {"dropout": 0.1}
    assert asdict(result)["final_loss"] == pytest.approx(0.3)
    assert result.metrics == {"accuracy": 0.88, "f1": 0.82}
    assert prediction.value == "BUY"
    assert prediction.confidence == pytest.approx(0.77)
    assert prediction.probabilities is not None
    assert np.allclose(prediction.probabilities, np.array([0.23, 0.77]))


def test_ensure_trained_raises_when_model_not_ready() -> None:
    """ensure_trained raises ModelNotTrainedError if model is not ready."""

    trainer = _ConcreteMLTrainer(trained=False)

    with pytest.raises(ModelNotTrainedError):
        trainer.ensure_trained()


def test_validate_input_shape_accepts_1d_and_2d_arrays() -> None:
    """validate_input_shape accepts non-empty 1D and 2D arrays."""

    trainer = _ConcreteMLTrainer(trained=True)
    trainer.validate_input_shape(np.array([1.0, 2.0], dtype=np.float64))
    trainer.validate_input_shape(np.array([[1.0, 2.0]], dtype=np.float64))


def test_validate_input_shape_rejects_invalid_inputs() -> None:
    """validate_input_shape rejects non-ndarray, empty, and unsupported dims."""

    trainer = _ConcreteMLTrainer(trained=True)

    with pytest.raises(TypeError):
        trainer.validate_input_shape([1.0, 2.0])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        trainer.validate_input_shape(np.array([], dtype=np.float64))
    with pytest.raises(ValueError):
        trainer.validate_input_shape(np.ones((1, 2, 3), dtype=np.float64))


def test_default_preprocess_and_postprocess_behaviour() -> None:
    """Default preprocess is identity and postprocess wraps prediction."""

    trainer = _ConcreteMLTrainer(trained=True)
    X = np.array([[1.0, 2.0]], dtype=np.float64)

    assert trainer.preprocess(X) is X
    wrapped = trainer.postprocess({"signal": "HOLD"})
    assert isinstance(wrapped, MLPrediction)
    assert wrapped.value == {"signal": "HOLD"}


def test_predict_supports_single_and_batch_inputs() -> None:
    """Concrete trainer predict returns single and batch prediction forms."""

    trainer = _ConcreteMLTrainer(trained=True)
    single = trainer.predict(np.array([1.0, 3.0], dtype=np.float64))
    batch = trainer.predict(np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float64))

    assert isinstance(single, MLPrediction)
    assert single.value == pytest.approx(2.0)
    assert isinstance(batch, list)
    assert len(batch) == 2
    assert all(isinstance(item, MLPrediction) for item in batch)
