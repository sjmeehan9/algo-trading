"""Unit tests for news sentiment MLTrainer implementation."""

from __future__ import annotations

import numpy as np
from algotrading.src.models.supporting.sentiment import (
    NewsSentimentTrainer,
    SentimentConfig,
    SentimentModelType,
)
from algotrading.src.trainers import MLPrediction, MLTrainingConfig


def test_trainer_train_and_predict_single_sample() -> None:
    trainer = NewsSentimentTrainer(
        config=SentimentConfig(model_type=SentimentModelType.VADER)
    )

    X = np.array(["Strong beat and raised guidance"], dtype=object)
    y = np.array([0.8], dtype=float)
    result = trainer.train(X, y, MLTrainingConfig(epochs=1))

    prediction = trainer.predict(X)
    assert result.epochs_trained == 1
    assert result.metrics is not None
    assert isinstance(prediction, MLPrediction)
    assert -1.0 <= float(prediction.value) <= 1.0


def test_trainer_predict_batch_and_probabilities() -> None:
    trainer = NewsSentimentTrainer(
        config=SentimentConfig(model_type=SentimentModelType.VADER)
    )
    X = np.array(["Positive", "Negative guidance"], dtype=object)
    y = np.array([0.6, -0.6], dtype=float)
    trainer.train(X, y, MLTrainingConfig())

    predictions = trainer.predict(X)
    probs = trainer.predict_proba(X)

    assert isinstance(predictions, list)
    assert len(predictions) == 2
    assert probs.shape == (2, 2)
    assert np.allclose(np.sum(probs, axis=1), np.ones(2))


def test_trainer_save_and_load_round_trip(tmp_path) -> None:
    trainer = NewsSentimentTrainer(
        config=SentimentConfig(model_type=SentimentModelType.VADER)
    )
    X = np.array(["Stable outlook"], dtype=object)
    y = np.array([0.1], dtype=float)
    trainer.train(X, y, MLTrainingConfig())

    target_dir = tmp_path / "sentiment-model"
    trainer.save(str(target_dir))

    loaded = NewsSentimentTrainer(
        config=SentimentConfig(model_type=SentimentModelType.VADER)
    )
    loaded.load(str(target_dir))
    prediction = loaded.predict(X)

    assert isinstance(prediction, MLPrediction)
