"""Configuration models for news sentiment supporting model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SentimentModelType(str, Enum):
    """Supported sentiment inference backends."""

    VADER = "vader"
    FINBERT = "finbert"
    PROVIDER_PASSTHROUGH = "provider"
    CUSTOM = "custom"


@dataclass(slots=True)
class SentimentConfig:
    """Configuration for sentiment inference and optional fine-tuning.

    Args:
        model_type: Backend type used for sentiment inference.
        model_name: Model identifier for transformer backends.
        max_length: Maximum processed text length in characters.
        batch_size: Batch size used during transformer inference.
        use_headline_only: Whether body text should be ignored.
        aggregate_method: Method for headline/body aggregation.
        cache_embeddings: Whether text sentiment should be cached.
        device: Device specifier for transformer backends.
    """

    model_type: SentimentModelType = SentimentModelType.VADER
    model_name: str = "ProsusAI/finbert"
    max_length: int = 512
    batch_size: int = 16
    use_headline_only: bool = False
    aggregate_method: str = "weighted"
    cache_embeddings: bool = True
    device: str = "cpu"

    def __post_init__(self) -> None:
        """Validate configuration invariants."""

        if self.max_length <= 0:
            raise ValueError("max_length must be greater than 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        if not self.model_name.strip():
            raise ValueError("model_name must be non-empty")
        if self.aggregate_method not in {"weighted", "concat"}:
            raise ValueError("aggregate_method must be either 'weighted' or 'concat'")


__all__ = ["SentimentModelType", "SentimentConfig"]
