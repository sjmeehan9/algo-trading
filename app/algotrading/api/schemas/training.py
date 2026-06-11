"""Request and response schemas for training control endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from algotrading.api.schemas.data_sources import validate_data_config_payload
from pydantic import BaseModel, Field, field_validator, model_validator


class TrainingJobStatus(str, Enum):
    """Lifecycle states for asynchronous training jobs."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_TERMINAL_STATUSES = frozenset(
    {
        TrainingJobStatus.COMPLETED,
        TrainingJobStatus.FAILED,
        TrainingJobStatus.CANCELLED,
    }
)


def is_terminal_status(status: TrainingJobStatus) -> bool:
    """Return True when the job has reached a terminal state."""

    return status in _TERMINAL_STATUSES


class TrainingJobCreate(BaseModel):
    """Payload to enqueue a new training job."""

    model_id: str = Field(min_length=1)
    training_config: dict[str, Any] = Field(default_factory=dict)
    data_config: dict[str, Any] = Field(default_factory=dict)
    total_timesteps: int | None = Field(default=None, ge=1)
    description: str | None = None
    # When set, warm-start training from this completed generation's saved
    # artifact (RL only) and train for ``total_timesteps`` additional steps,
    # instead of initialising a fresh model. ``None`` keeps from-scratch.
    continue_from_generation_id: str | None = None

    @field_validator("continue_from_generation_id", mode="before")
    @classmethod
    def _normalize_continue_generation(cls, value: object) -> str | None:
        """Treat blank continue-from values as unset (from-scratch)."""

        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("data_config")
    @classmethod
    def _validate_data_config(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Validate loose job data-source overrides when present."""

        return validate_data_config_payload(value, require_complete=False)

    @model_validator(mode="after")
    def _validate_payload(self) -> "TrainingJobCreate":
        """Validate cross-field constraints on the create payload."""

        if not self.model_id.strip():
            raise ValueError("model_id is required")
        return self


class TrainingQueueReorderRequest(BaseModel):
    """Payload for replacing the queued-job execution order."""

    job_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_unique_job_ids(self) -> "TrainingQueueReorderRequest":
        """Validate that each queued job ID appears once."""

        normalized_ids = [job_id.strip() for job_id in self.job_ids]
        if any(not job_id for job_id in normalized_ids):
            raise ValueError("job_ids cannot contain blank values")
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("job_ids must be unique")
        self.job_ids = normalized_ids
        return self


class TrainingJob(BaseModel):
    """API representation of a training job and its current state."""

    job_id: str
    model_id: str
    status: TrainingJobStatus
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    progress_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    current_timestep: int = Field(default=0, ge=0)
    total_timesteps: int = Field(default=0, ge=0)
    current_metrics: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None
    generation_id: str | None = None
    description: str | None = None
    training_config: dict[str, Any] = Field(default_factory=dict)
    data_config: dict[str, Any] = Field(default_factory=dict)
    continue_from_generation_id: str | None = None


class TrainingProgress(BaseModel):
    """Snapshot of training progress broadcast over WebSocket."""

    job_id: str
    model_id: str
    status: TrainingJobStatus
    progress_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    current_timestep: int = Field(default=0, ge=0)
    total_timesteps: int = Field(default=0, ge=0)
    current_metrics: dict[str, Any] = Field(default_factory=dict)
    estimated_remaining_seconds: float | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


__all__ = [
    "TrainingJobStatus",
    "TrainingJobCreate",
    "TrainingQueueReorderRequest",
    "TrainingJob",
    "TrainingProgress",
    "is_terminal_status",
]
