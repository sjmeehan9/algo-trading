"""Request and response schemas for pre-deployment model selection."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from algotrading.api.schemas.backtesting import PerformanceMetrics
from pydantic import BaseModel, Field


class ReadinessCheckStatus(str, Enum):
    """Possible outcomes for one deployment readiness criterion."""

    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


class ReadinessCheck(BaseModel):
    """One readiness criterion returned by deployment validation."""

    key: str
    label: str
    status: ReadinessCheckStatus
    message: str


class DeploymentReadiness(BaseModel):
    """Deployability verdict for a selected model generation."""

    deployable: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    checks: list[ReadinessCheck] = Field(default_factory=list)


class DeploymentValidationRequest(BaseModel):
    """Payload used to validate a deployment candidate selection."""

    model_id: str = Field(min_length=1)
    generation_id: str = Field(min_length=1)


class DeploymentBacktestSummary(BaseModel):
    """Latest evaluation/backtest evidence for one generation."""

    source: str
    backtest_id: str | None = None
    completed_at: datetime | None = None
    metrics: PerformanceMetrics | dict[str, float | int | None]


class DeploymentGenerationSummary(BaseModel):
    """Generation metadata displayed by the deployment workflow."""

    generation_id: str
    generation_number: int
    status: str
    created_at: datetime
    training_duration_seconds: float
    final_reward: float | None = None
    model_path: str | None = None
    metrics: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    evaluation: DeploymentBacktestSummary | None = None


class DeploymentCandidate(BaseModel):
    """Core RL model candidate plus generation and readiness metadata."""

    model_id: str
    name: str
    description: str | None = None
    algorithm: str
    trainer_type: str
    state: str
    supporting_model_ids: list[str] = Field(default_factory=list)
    strategy_ids: list[str] = Field(default_factory=list)
    available_generations: list[DeploymentGenerationSummary] = Field(
        default_factory=list
    )
    latest_backtest: DeploymentBacktestSummary | None = None
    readiness: DeploymentReadiness


class DeploymentSelection(BaseModel):
    """Persisted pre-deployment candidate selection consumed by later phases."""

    model_id: str
    generation_id: str
    selected_at: datetime
    candidate: DeploymentCandidate
    readiness: DeploymentReadiness
    selected_by: str | None = None


__all__ = [
    "ReadinessCheckStatus",
    "ReadinessCheck",
    "DeploymentReadiness",
    "DeploymentValidationRequest",
    "DeploymentBacktestSummary",
    "DeploymentGenerationSummary",
    "DeploymentCandidate",
    "DeploymentSelection",
]
