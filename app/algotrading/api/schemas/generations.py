"""Request and response schemas for generation-history endpoints."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class GenerationSummary(BaseModel):
    """Compact generation information used in list responses."""

    generation_id: str
    generation_number: int
    model_id: str
    created_at: datetime
    training_duration_seconds: float
    final_reward: float | None = None
    metrics: dict[str, float | int | str | bool | None] = Field(default_factory=dict)


class GenerationDetail(GenerationSummary):
    """Extended generation payload with training metadata."""

    hyperparameters: dict[str, object] = Field(default_factory=dict)
    training_config: dict[str, object] = Field(default_factory=dict)
    metrics_history: list[dict[str, object]] = Field(default_factory=list)
    model_path: str | None = None


class GenerationComparison(BaseModel):
    """Result of comparing multiple generations for one model."""

    generations: list[GenerationSummary]
    metric_deltas: dict[str, dict[str, float | None]] = Field(default_factory=dict)


class GenerationComparisonRequest(BaseModel):
    """Comparison request payload for selected generation IDs."""

    model_id: str = Field(min_length=1)
    generation_ids: list[str] = Field(min_length=2)


__all__ = [
    "GenerationSummary",
    "GenerationDetail",
    "GenerationComparison",
    "GenerationComparisonRequest",
]
