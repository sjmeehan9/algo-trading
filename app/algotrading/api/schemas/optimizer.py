"""Request and response schemas for hyperparameter optimization endpoints."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from algotrading.api.schemas.models import ModelConfigResponse
from pydantic import BaseModel, Field

HyperparameterPrimitive = int | float | str | bool | None
HyperparameterValue = int | float | str | bool


class SuggestionConfidence(str, Enum):
    """Confidence levels assigned to optimizer suggestions."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class OptimizationSource(str, Enum):
    """Source used to generate an optimization result."""

    OPENAI = "openai"
    FALLBACK = "fallback"


class SuggestionOutcomeStatus(str, Enum):
    """Lifecycle state for applied suggestion outcome tracking."""

    NOT_APPLIED = "not_applied"
    PENDING_NEXT_GENERATION = "pending_next_generation"
    EVALUATED = "evaluated"


class OptimizerAnalyzeRequest(BaseModel):
    """Request payload for analyzing model training history."""

    model_id: str = Field(min_length=1)
    max_generations: int = Field(default=5, ge=1, le=20)
    include_backtest_metrics: bool = True


class HyperparameterSuggestion(BaseModel):
    """One actionable hyperparameter adjustment suggestion."""

    suggestion_id: str
    parameter: str = Field(min_length=1)
    current_value: HyperparameterPrimitive = None
    suggested_value: HyperparameterValue
    rationale: str = Field(min_length=1)
    confidence: SuggestionConfidence = SuggestionConfidence.MEDIUM
    expected_impact: str = Field(min_length=1)
    applied: bool = False
    applied_at: datetime | None = None
    outcome_status: SuggestionOutcomeStatus = SuggestionOutcomeStatus.NOT_APPLIED
    outcome_notes: str | None = None


class OptimizationResult(BaseModel):
    """Stored optimizer analysis output for one model."""

    result_id: str
    model_id: str
    created_at: datetime
    analyzed_generation_ids: list[str] = Field(default_factory=list)
    suggestions: list[HyperparameterSuggestion] = Field(default_factory=list)
    priority_changes: list[str] = Field(default_factory=list)
    summary: str
    source: OptimizationSource
    raw_response: str | None = None


class ApplySuggestionRequest(BaseModel):
    """Request payload for applying one stored suggestion to a model."""

    model_id: str = Field(min_length=1)
    suggestion_id: str = Field(min_length=1)
    result_id: str | None = None


class AppliedSuggestionResponse(BaseModel):
    """Response returned after applying a suggestion to model config."""

    suggestion: HyperparameterSuggestion
    updated_model: ModelConfigResponse
    optimization_result: OptimizationResult


__all__ = [
    "AppliedSuggestionResponse",
    "ApplySuggestionRequest",
    "HyperparameterPrimitive",
    "HyperparameterValue",
    "HyperparameterSuggestion",
    "OptimizationResult",
    "OptimizationSource",
    "OptimizerAnalyzeRequest",
    "SuggestionConfidence",
    "SuggestionOutcomeStatus",
]
