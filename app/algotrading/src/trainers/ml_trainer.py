"""Abstract interface for non-reinforcement machine learning trainers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from algotrading.src.trainers.exceptions import ModelNotTrainedError


@dataclass(slots=True)
class MLTrainingConfig:
    """Configuration for ML training runs.

    Args:
        epochs: Optional number of training epochs.
        batch_size: Optional batch size.
        validation_split: Optional validation split ratio in the range [0, 1).
        early_stopping: Whether early stopping is enabled.
        custom_params: Optional framework-specific hyperparameters.
    """

    epochs: int | None = None
    batch_size: int | None = None
    validation_split: float | None = None
    early_stopping: bool = False
    custom_params: dict[str, object] | None = None


@dataclass(slots=True)
class MLTrainingResult:
    """Outcome metadata from a completed ML training run.

    Args:
        epochs_trained: Number of completed epochs.
        final_loss: Final training loss value.
        validation_loss: Optional final validation loss value.
        training_time_seconds: Total elapsed training time.
        model_path: Optional persisted model path.
        metrics: Optional additional metrics such as accuracy/f1.
    """

    epochs_trained: int
    final_loss: float
    validation_loss: float | None
    training_time_seconds: float
    model_path: str | None = None
    metrics: dict[str, float] | None = None


@dataclass(slots=True)
class MLPrediction:
    """Standardized prediction payload for ML inference.

    Args:
        value: Model prediction value for single-sample inference.
        confidence: Optional confidence score for the prediction.
        probabilities: Optional class probabilities for classifiers.
    """

    value: Any
    confidence: float | None = None
    probabilities: np.ndarray | None = None


class MLTrainer(ABC):
    """Abstract lifecycle interface for ML trainer implementations.

    Implementations are responsible for creating, training, saving/loading,
    and running inference with non-RL machine learning models.
    """

    @abstractmethod
    def create_model(
        self,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        config: MLTrainingConfig,
    ) -> None:
        """Initialize a model architecture.

        Args:
            input_shape: Expected input shape for training/inference.
            output_shape: Expected output shape for predictions.
            config: ML model configuration and hyperparameters.
        """

        raise NotImplementedError

    @abstractmethod
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: MLTrainingConfig,
    ) -> MLTrainingResult:
        """Train the model on feature and target arrays.

        Args:
            X: Feature matrix.
            y: Target array.
            config: Training configuration.

        Returns:
            Aggregated training result metadata.
        """

        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> MLPrediction | list[MLPrediction]:
        """Infer predictions for one or more samples.

        Args:
            X: Input samples, either single-sample or batch.

        Returns:
            A single prediction for one-sample input or a list for batches.
        """

        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Infer class probabilities for classification models.

        Args:
            X: Input samples.

        Returns:
            Probability distribution array for each sample.
        """

        raise NotImplementedError

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Persist model artifacts to disk.

        Args:
            filepath: Destination path for persisted model data.
        """

        raise NotImplementedError

    @abstractmethod
    def load(self, filepath: str) -> None:
        """Load persisted model artifacts from disk.

        Args:
            filepath: Source path for persisted model data.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return a model type identifier."""

        raise NotImplementedError

    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """Return whether the model is ready for inference."""

        raise NotImplementedError

    def ensure_trained(self) -> None:
        """Ensure that inference-dependent actions are valid.

        Raises:
            ModelNotTrainedError: If the model has not been trained or loaded.
        """

        if not self.is_trained:
            raise ModelNotTrainedError(
                message="Model has not been trained or loaded.",
                trainer_name=self.model_type,
            )

    def validate_input_shape(self, X: np.ndarray) -> None:
        """Validate ML input array shape.

        Args:
            X: Input array to validate.

        Raises:
            ValueError: If input array is empty or has unsupported dimensionality.
            TypeError: If input is not a numpy ndarray.
        """

        if not isinstance(X, np.ndarray):
            raise TypeError(f"Input must be a numpy.ndarray, got {type(X).__name__}.")

        if X.size == 0:
            raise ValueError("Input array must not be empty.")

        if X.ndim not in (1, 2):
            raise ValueError(
                "Input array must be 1D (single sample) or 2D (batch samples)."
            )

    def preprocess(self, X: np.ndarray) -> np.ndarray:
        """Preprocess inputs prior to training/inference.

        Default behavior returns the input unchanged.

        Args:
            X: Input array.

        Returns:
            Preprocessed input array.
        """

        return X

    def postprocess(self, prediction: Any) -> MLPrediction:
        """Convert raw model output into a standardized prediction payload.

        Args:
            prediction: Raw model output.

        Returns:
            Wrapped prediction payload.
        """

        return MLPrediction(value=prediction)


__all__ = [
    "MLTrainer",
    "MLTrainingConfig",
    "MLTrainingResult",
    "MLPrediction",
]
