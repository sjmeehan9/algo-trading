"""Unit tests for news sentiment model inference paths."""

from __future__ import annotations

from datetime import UTC, datetime

from algotrading.src.data_pipeline import NewsRecord
from algotrading.src.models.signals import SignalType
from algotrading.src.models.supporting.sentiment import (
    NewsSentimentModel,
    SentimentConfig,
    SentimentModelType,
)


def test_vader_predict_returns_sentiment_signal() -> None:
    model = NewsSentimentModel(SentimentConfig(model_type=SentimentModelType.VADER))

    signal = model.predict("Company beats earnings estimates and raises guidance")

    assert signal.signal_type == SignalType.SENTIMENT
    assert -1.0 <= signal.value <= 1.0
    assert signal.confidence is not None
    assert 0.0 <= signal.confidence <= 1.0


def test_predict_batch_returns_one_signal_per_text() -> None:
    model = NewsSentimentModel(SentimentConfig(model_type=SentimentModelType.VADER))

    signals = model.predict_batch(
        [
            "Strong quarter with improved margins",
            "Guidance lowered due to demand weakness",
        ]
    )

    assert len(signals) == 2
    assert all(signal.signal_type == SignalType.SENTIMENT for signal in signals)


def test_analyze_news_uses_provider_fast_path() -> None:
    model = NewsSentimentModel(SentimentConfig(model_type=SentimentModelType.VADER))
    news = NewsRecord(
        timestamp=datetime.now(tz=UTC),
        headline="Provider has precomputed score",
        body=None,
        source="alpha-vantage",
        symbols=["AAPL"],
        categories=["earnings"],
        sentiment_score=0.72,
        url=None,
        news_id="n-1",
    )

    signal = model.analyze_news(news)

    assert signal.value == 0.72
    assert signal.metadata.model_type == "provider_passthrough"
    assert signal.symbol == "AAPL"


def test_analyze_news_runs_local_inference_when_provider_score_missing() -> None:
    model = NewsSentimentModel(SentimentConfig(model_type=SentimentModelType.VADER))
    news = NewsRecord(
        timestamp=datetime.now(tz=UTC),
        headline="Upbeat outlook from management",
        body="Revenue growth accelerated quarter over quarter.",
        source="benzinga",
        symbols=["MSFT"],
        categories=["guidance"],
        sentiment_score=None,
        url=None,
        news_id="n-2",
    )

    signal = model.analyze_news(news)

    assert signal.signal_type == SignalType.SENTIMENT
    assert signal.symbol == "MSFT"
    assert signal.metadata.model_type == "ml"


def test_finbert_branch_is_supported_via_batched_scoring(monkeypatch) -> None:
    def _noop_load_backend(self) -> None:  # noqa: ANN001
        return

    def _fake_scores(self, processed_texts):  # noqa: ANN001
        return [(0.6, 0.8) for _ in processed_texts]

    monkeypatch.setattr(NewsSentimentModel, "_load_model_backend", _noop_load_backend)
    monkeypatch.setattr(NewsSentimentModel, "_predict_finbert_scores", _fake_scores)

    model = NewsSentimentModel(SentimentConfig(model_type=SentimentModelType.FINBERT))
    signals = model.predict_batch(["Bullish setup", "Bearish setup"])

    assert len(signals) == 2
    assert all(signal.value == 0.6 for signal in signals)
    assert all(signal.confidence == 0.8 for signal in signals)
