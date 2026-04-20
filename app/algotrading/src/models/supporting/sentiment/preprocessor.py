"""Text preprocessing utilities for sentiment inference."""

from __future__ import annotations

import re

from algotrading.src.models.supporting.sentiment.config import SentimentConfig

_URL_REGEX = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_SYMBOL_REGEX = re.compile(r"\$([A-Za-z][A-Za-z0-9._-]{0,9})")
_WHITESPACE_REGEX = re.compile(r"\s+")
_DISALLOWED_CHAR_REGEX = re.compile(r"[^A-Za-z0-9\s\.,:;!?%+\-\[\]\(\)/]")


class TextPreprocessor:
    """Normalize and compose news text for downstream sentiment models."""

    def __init__(self, config: SentimentConfig) -> None:
        """Initialize preprocessor with sentiment configuration."""

        self._config = config

    def preprocess(self, text: str) -> str:
        """Clean text for sentiment model consumption.

        Args:
            text: Raw input text.

        Returns:
            Cleaned and normalized text truncated to configured max length.
        """

        if not text.strip():
            return ""

        cleaned = _URL_REGEX.sub(" ", text)
        cleaned = _SYMBOL_REGEX.sub(r"\1", cleaned)
        cleaned = _DISALLOWED_CHAR_REGEX.sub(" ", cleaned)
        cleaned = _WHITESPACE_REGEX.sub(" ", cleaned).strip()
        if len(cleaned) > self._config.max_length:
            return cleaned[: self._config.max_length].rstrip()
        return cleaned

    def prepare_for_model(self, headline: str, body: str | None = None) -> str:
        """Compose normalized model input from headline and optional body text."""

        normalized_headline = self.preprocess(headline)
        if self._config.use_headline_only or not body:
            return normalized_headline

        normalized_body = self.preprocess(body)
        if not normalized_body:
            return normalized_headline

        if self._config.aggregate_method == "weighted":
            prefix = f"[HEADLINE] {normalized_headline} [BODY] "
            remaining = max(self._config.max_length - len(prefix), 0)
            body_segment = normalized_body[:remaining]
            return f"{prefix}{body_segment}".strip()

        merged = f"{normalized_headline} {normalized_body}".strip()
        return merged[: self._config.max_length].rstrip()

    def batch_preprocess(self, texts: list[str]) -> list[str]:
        """Preprocess a list of text inputs preserving order."""

        return [self.preprocess(item) for item in texts]


__all__ = ["TextPreprocessor"]
