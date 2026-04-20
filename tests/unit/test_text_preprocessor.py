"""Unit tests for sentiment text preprocessor."""

from __future__ import annotations

from algotrading.src.models.supporting.sentiment import (
    SentimentConfig,
    TextPreprocessor,
)


def test_preprocess_removes_urls_symbols_and_normalizes_whitespace() -> None:
    config = SentimentConfig(max_length=120)
    preprocessor = TextPreprocessor(config)

    processed = preprocessor.preprocess(
        "  Bullish update on $AAPL at https://example.com/news?id=1   "
    )

    assert "http" not in processed.lower()
    assert "$AAPL" not in processed
    assert "AAPL" in processed
    assert "  " not in processed


def test_prepare_for_model_headline_only() -> None:
    config = SentimentConfig(use_headline_only=True)
    preprocessor = TextPreprocessor(config)

    combined = preprocessor.prepare_for_model(
        headline="Strong earnings beat",
        body="Body should not be included",
    )

    assert "HEADLINE" not in combined
    assert "Body" not in combined
    assert "Strong earnings beat" in combined


def test_prepare_for_model_weighted_combines_headline_and_body() -> None:
    config = SentimentConfig(use_headline_only=False, aggregate_method="weighted")
    preprocessor = TextPreprocessor(config)

    combined = preprocessor.prepare_for_model(
        headline="Fed signals pause",
        body="Risk assets rallied after the announcement.",
    )

    assert combined.startswith("[HEADLINE]")
    assert "[BODY]" in combined


def test_batch_preprocess_preserves_order() -> None:
    config = SentimentConfig(max_length=50)
    preprocessor = TextPreprocessor(config)

    values = preprocessor.batch_preprocess(["First  ", "Second", "Third"])
    assert values == ["First", "Second", "Third"]
