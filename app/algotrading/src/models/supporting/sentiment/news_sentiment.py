"""News sentiment supporting model and MLTrainer integration."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from algotrading.src.data_pipeline import NewsRecord
from algotrading.src.models.signals import ModelSignal, SignalMetadata, SignalType
from algotrading.src.models.supporting.sentiment.config import (
    SentimentConfig,
    SentimentModelType,
)
from algotrading.src.models.supporting.sentiment.preprocessor import TextPreprocessor
from algotrading.src.trainers import (
    MLPrediction,
    MLTrainer,
    MLTrainingConfig,
    MLTrainingResult,
)


class NewsSentimentModel:
    """Sentiment model that produces standardized sentiment signals from news text."""

    def __init__(
        self,
        config: SentimentConfig | None = None,
        model_id: str = "news_sentiment_model",
    ) -> None:
        """Initialize model backend and preprocessing pipeline."""

        self._config = config or SentimentConfig()
        self._model_id = model_id
        self._preprocessor = TextPreprocessor(self._config)
        self._cache: dict[str, tuple[float, float]] = {}
        self._calibration_bias = 0.0

        self._vader_analyzer: Any | None = None
        self._tokenizer: Any | None = None
        self._transformer_model: Any | None = None
        self._torch: Any | None = None

        self._load_model_backend()

    @property
    def config(self) -> SentimentConfig:
        """Return model configuration."""

        return self._config

    @property
    def model_id(self) -> str:
        """Return model identifier used in signal metadata."""

        return self._model_id

    @property
    def calibration_bias(self) -> float:
        """Return additive calibration bias learned from labeled data."""

        return self._calibration_bias

    def predict(
        self,
        text: str,
        timestamp: datetime | None = None,
        symbol: str | None = None,
    ) -> ModelSignal:
        """Analyze sentiment of a single text input and return one signal."""

        inference_start = time.perf_counter()
        processed = self._preprocessor.preprocess(text)

        if not processed:
            value = 0.0
            confidence = 0.0
        else:
            value, confidence = self._predict_scores(processed)

        elapsed_ms = (time.perf_counter() - inference_start) * 1000
        resolved_timestamp = timestamp or datetime.now(tz=UTC)
        if resolved_timestamp.tzinfo is None or resolved_timestamp.utcoffset() is None:
            resolved_timestamp = resolved_timestamp.replace(tzinfo=UTC)

        return ModelSignal(
            timestamp=resolved_timestamp,
            signal_type=SignalType.SENTIMENT,
            value=value,
            confidence=confidence,
            symbol=symbol,
            metadata=SignalMetadata(
                model_id=self._model_id,
                model_type="ml",
                inference_time_ms=elapsed_ms,
                extra={"backend": self._config.model_type.value},
            ),
        )

    def predict_batch(
        self,
        texts: list[str],
        timestamps: list[datetime] | None = None,
        symbols: list[str | None] | None = None,
    ) -> list[ModelSignal]:
        """Run efficient sentiment inference for a batch of text inputs."""

        if not texts:
            return []

        if timestamps is not None and len(timestamps) != len(texts):
            raise ValueError("timestamps length must match texts length")
        if symbols is not None and len(symbols) != len(texts):
            raise ValueError("symbols length must match texts length")

        processed = self._preprocessor.batch_preprocess(texts)
        now = datetime.now(tz=UTC)
        batched_start = time.perf_counter()

        if self._config.model_type in {
            SentimentModelType.FINBERT,
            SentimentModelType.CUSTOM,
        }:
            scored = self._predict_finbert_scores(processed)
        else:
            scored = [self._predict_scores(text_value) for text_value in processed]

        elapsed_ms = (time.perf_counter() - batched_start) * 1000
        signals: list[ModelSignal] = []
        for index, (value, confidence) in enumerate(scored):
            signal_timestamp = timestamps[index] if timestamps else now
            if signal_timestamp.tzinfo is None or signal_timestamp.utcoffset() is None:
                signal_timestamp = signal_timestamp.replace(tzinfo=UTC)

            signals.append(
                ModelSignal(
                    timestamp=signal_timestamp,
                    signal_type=SignalType.SENTIMENT,
                    value=value,
                    confidence=confidence,
                    symbol=symbols[index] if symbols else None,
                    metadata=SignalMetadata(
                        model_id=self._model_id,
                        model_type="ml",
                        inference_time_ms=elapsed_ms / len(scored),
                        extra={"backend": self._config.model_type.value},
                    ),
                )
            )
        return signals

    def analyze_news(self, news: NewsRecord) -> ModelSignal:
        """Analyze one NewsRecord using provider fast-path or local inference."""

        symbol = news.symbols[0] if news.symbols else None
        if news.sentiment_score is not None:
            return ModelSignal(
                timestamp=news.timestamp,
                signal_type=SignalType.SENTIMENT,
                value=float(news.sentiment_score),
                confidence=0.5,
                symbol=symbol,
                metadata=SignalMetadata(
                    model_id=self._model_id,
                    model_type="provider_passthrough",
                    inference_time_ms=0.0,
                    extra={"source": news.source, "news_id": news.news_id},
                ),
            )

        text = self._preprocessor.prepare_for_model(news.headline, news.body)
        return self.predict(text=text, timestamp=news.timestamp, symbol=symbol)

    def prepare_text(self, headline: str, body: str | None = None) -> str:
        """Prepare headline/body payload into normalized model input text."""

        return self._preprocessor.prepare_for_model(headline=headline, body=body)

    def fit_calibration(
        self, texts: list[str], labels: list[float]
    ) -> dict[str, float]:
        """Fit a lightweight additive calibration bias on labeled sentiment data."""

        if len(texts) != len(labels):
            raise ValueError("texts and labels must have matching lengths")
        if not texts:
            raise ValueError("texts must be non-empty")

        predictions = np.array([self._predict_scores(t)[0] for t in texts], dtype=float)
        target = np.clip(np.array(labels, dtype=float), -1.0, 1.0)
        residuals = target - predictions
        bias = float(np.clip(np.mean(residuals), -1.0, 1.0))
        self._calibration_bias = bias

        calibrated = np.clip(predictions + bias, -1.0, 1.0)
        mse = float(np.mean((target - calibrated) ** 2))
        mae = float(np.mean(np.abs(target - calibrated)))
        return {"calibration_bias": bias, "mse": mse, "mae": mae}

    def save(self, filepath: str) -> None:
        """Persist model configuration and calibration state to disk."""

        path = Path(filepath)
        if path.suffix:
            target = path
            target.parent.mkdir(parents=True, exist_ok=True)
            base_dir = target.parent
        else:
            base_dir = path
            base_dir.mkdir(parents=True, exist_ok=True)
            target = base_dir / "news_sentiment.json"

        payload = {
            "model_id": self._model_id,
            "config": {
                **asdict(self._config),
                "model_type": self._config.model_type.value,
            },
            "calibration_bias": self._calibration_bias,
        }
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

        if (
            self._config.model_type
            in {SentimentModelType.FINBERT, SentimentModelType.CUSTOM}
            and self._transformer_model is not None
            and self._tokenizer is not None
            and hasattr(self._transformer_model, "save_pretrained")
            and hasattr(self._tokenizer, "save_pretrained")
        ):
            model_dir = base_dir / "transformer"
            model_dir.mkdir(parents=True, exist_ok=True)
            self._transformer_model.save_pretrained(str(model_dir))
            self._tokenizer.save_pretrained(str(model_dir))

    @classmethod
    def load(cls, filepath: str) -> NewsSentimentModel:
        """Load persisted model state from disk."""

        path = Path(filepath)
        if path.is_dir():
            source = path / "news_sentiment.json"
        else:
            source = path

        payload = json.loads(source.read_text(encoding="utf-8"))
        config_data = dict(payload["config"])
        config_data["model_type"] = SentimentModelType(str(config_data["model_type"]))
        config = SentimentConfig(**config_data)
        model = cls(
            config=config, model_id=str(payload.get("model_id", "news_sentiment_model"))
        )
        model._calibration_bias = float(payload.get("calibration_bias", 0.0))
        return model

    def _load_model_backend(self) -> None:
        """Initialize model backend according to configured model type."""

        if self._config.model_type == SentimentModelType.PROVIDER_PASSTHROUGH:
            return

        if self._config.model_type == SentimentModelType.VADER:
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            except ImportError as exc:  # pragma: no cover - environment-specific
                raise RuntimeError(
                    "vaderSentiment is required for SentimentModelType.VADER"
                ) from exc
            self._vader_analyzer = SentimentIntensityAnalyzer()
            return

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "transformers and torch are required for FINBERT/CUSTOM model types"
            ) from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self._config.model_name)
        self._transformer_model = AutoModelForSequenceClassification.from_pretrained(
            self._config.model_name
        )
        if self._config.device == "cuda" and torch.cuda.is_available():
            self._transformer_model = self._transformer_model.to("cuda")
        self._transformer_model.eval()

    def _predict_scores(self, processed_text: str) -> tuple[float, float]:
        """Compute sentiment score and confidence for one preprocessed text."""

        if not processed_text:
            return 0.0, 0.0

        if self._config.cache_embeddings and processed_text in self._cache:
            return self._cache[processed_text]

        if self._config.model_type in {
            SentimentModelType.FINBERT,
            SentimentModelType.CUSTOM,
        }:
            value, confidence = self._predict_finbert_scores([processed_text])[0]
        elif self._config.model_type == SentimentModelType.PROVIDER_PASSTHROUGH:
            value, confidence = 0.0, 0.0
        else:
            if self._vader_analyzer is None:
                raise RuntimeError("VADER backend is not initialized")
            scores = self._vader_analyzer.polarity_scores(processed_text)
            value = self._apply_calibration(float(scores["compound"]))
            confidence = float(min(1.0, max(0.0, abs(value))))

        result = (value, confidence)
        if self._config.cache_embeddings:
            self._cache[processed_text] = result
        return result

    def _predict_finbert_scores(
        self,
        processed_texts: list[str],
    ) -> list[tuple[float, float]]:
        """Compute sentiment score and confidence for transformer backends."""

        if (
            self._tokenizer is None
            or self._transformer_model is None
            or self._torch is None
        ):
            raise RuntimeError("Transformer backend is not initialized")

        filtered = [text if text else "neutral" for text in processed_texts]
        inputs = self._tokenizer(
            filtered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._config.max_length,
        )

        if self._config.device == "cuda" and self._torch.cuda.is_available():
            inputs = {key: value.to("cuda") for key, value in inputs.items()}

        with self._torch.no_grad():
            outputs = self._transformer_model(**inputs)
            probabilities = self._torch.softmax(outputs.logits, dim=-1)

        values: list[tuple[float, float]] = []
        for neg, _neu, pos in probabilities.tolist():
            sentiment_value = self._apply_calibration(float(pos) - float(neg))
            confidence = float(min(1.0, max(0.0, max(float(pos), float(neg)))))
            values.append((sentiment_value, confidence))
        return values

    def _apply_calibration(self, value: float) -> float:
        """Apply additive calibration while preserving output range constraints."""

        adjusted = value + self._calibration_bias
        return float(max(-1.0, min(1.0, adjusted)))


class NewsSentimentTrainer(MLTrainer):
    """MLTrainer wrapper around NewsSentimentModel with optional calibration fit."""

    def __init__(
        self,
        config: SentimentConfig | None = None,
        model_id: str = "news_sentiment_model",
    ) -> None:
        """Initialize trainer with a preloaded sentiment model backend."""

        self._config = config or SentimentConfig()
        self._model = NewsSentimentModel(config=self._config, model_id=model_id)
        self._trained = True

    def create_model(
        self,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        config: MLTrainingConfig,
    ) -> None:
        """Initialize model from training config, preserving default fallbacks."""

        del input_shape, output_shape
        custom = config.custom_params or {}
        candidate_type = custom.get("model_type", self._config.model_type.value)
        model_type = (
            candidate_type
            if isinstance(candidate_type, SentimentModelType)
            else SentimentModelType(str(candidate_type))
        )
        self._config = SentimentConfig(
            model_type=model_type,
            model_name=str(custom.get("model_name", self._config.model_name)),
            max_length=int(custom.get("max_length", self._config.max_length)),
            batch_size=int(custom.get("batch_size", self._config.batch_size)),
            use_headline_only=bool(
                custom.get("use_headline_only", self._config.use_headline_only)
            ),
            aggregate_method=str(
                custom.get("aggregate_method", self._config.aggregate_method)
            ),
            cache_embeddings=bool(
                custom.get("cache_embeddings", self._config.cache_embeddings)
            ),
            device=str(custom.get("device", self._config.device)),
        )
        self._model = NewsSentimentModel(
            config=self._config, model_id=self._model.model_id
        )
        self._trained = True

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: MLTrainingConfig,
    ) -> MLTrainingResult:
        """Fit calibration parameters on labeled sentiment samples."""

        start = time.perf_counter()
        self.validate_input_shape(X)
        self.validate_input_shape(y)
        texts = self._extract_texts(X)
        labels = self._extract_labels(y)
        metrics = self._model.fit_calibration(texts=texts, labels=labels)
        self._trained = True
        elapsed_seconds = time.perf_counter() - start

        epochs = config.epochs if config.epochs is not None else 1
        return MLTrainingResult(
            epochs_trained=max(1, int(epochs)),
            final_loss=metrics["mse"],
            validation_loss=None,
            training_time_seconds=elapsed_seconds,
            model_path=None,
            metrics=metrics,
        )

    def predict(self, X: np.ndarray | object) -> MLPrediction | list[MLPrediction]:
        """Infer sentiment predictions for one or more text samples."""

        self.ensure_trained()
        payload = self._normalize_payload(X)
        items = self._extract_texts(payload)
        if len(items) == 1:
            signal = self._model.predict(items[0])
            return MLPrediction(value=signal.value, confidence=signal.confidence)

        signals = self._model.predict_batch(items)
        return [
            MLPrediction(value=signal.value, confidence=signal.confidence)
            for signal in signals
        ]

    def predict_proba(self, X: np.ndarray | object) -> np.ndarray:
        """Return two-class probabilities derived from sentiment scale values."""

        prediction = self.predict(X)
        if isinstance(prediction, list):
            values = [float(item.value) for item in prediction]
        else:
            values = [float(prediction.value)]

        rows: list[list[float]] = []
        for value in values:
            positive = min(1.0, max(0.0, (value + 1.0) / 2.0))
            rows.append([1.0 - positive, positive])
        return np.array(rows, dtype=np.float64)

    def save(self, filepath: str) -> None:
        """Persist trainer model state to disk."""

        self._model.save(filepath)

    def load(self, filepath: str) -> None:
        """Load trainer model state from disk."""

        self._model = NewsSentimentModel.load(filepath)
        self._config = self._model.config
        self._trained = True

    @property
    def model_type(self) -> str:
        """Return trainer type identifier."""

        return "news_sentiment"

    @property
    def is_trained(self) -> bool:
        """Return whether trainer is ready to perform inference."""

        return self._trained

    @property
    def sentiment_model(self) -> NewsSentimentModel:
        """Expose underlying sentiment model for integration consumers."""

        return self._model

    def predict_news(self, news: NewsRecord) -> ModelSignal:
        """Convenience helper to infer directly from NewsRecord."""

        self.ensure_trained()
        return self._model.analyze_news(news)

    def _extract_texts(self, payload: np.ndarray) -> list[str]:
        """Normalize diverse payload inputs into ordered text samples."""

        if payload.ndim == 0:
            value = payload.item()
            return [self._coerce_payload_to_text(value)]

        if payload.ndim == 1:
            return [self._coerce_payload_to_text(item) for item in payload.tolist()]

        texts: list[str] = []
        for row in payload:
            if row.size == 1:
                texts.append(self._coerce_payload_to_text(row[0]))
            else:
                joined = " ".join(self._coerce_payload_to_text(item) for item in row)
                texts.append(joined)
        return texts

    def _normalize_payload(self, payload: np.ndarray | object) -> np.ndarray:
        """Normalize arbitrary inference payload values into numpy arrays."""

        if isinstance(payload, np.ndarray):
            return payload
        return np.array([payload], dtype=object)

    def _extract_labels(self, values: np.ndarray) -> list[float]:
        """Normalize labels to clipped floating sentiment scale values."""

        flattened = values.reshape(-1).tolist()
        labels = [float(item) for item in flattened]
        if any(not math.isfinite(item) for item in labels):
            raise ValueError("All labels must be finite numbers")
        return [float(max(-1.0, min(1.0, label))) for label in labels]

    def _coerce_payload_to_text(self, value: object) -> str:
        """Coerce model payload item into text for sentiment analysis."""

        if isinstance(value, str):
            return value

        if isinstance(value, dict):
            headline = str(value.get("headline", "")).strip()
            body_value = value.get("body")
            body = str(body_value).strip() if body_value is not None else None
            if headline or body:
                return self._model.prepare_text(
                    headline=headline,
                    body=body,
                )
            return str(value)

        return str(value)


__all__ = ["NewsSentimentModel", "NewsSentimentTrainer"]
