"""Request and response schemas for supporting model lifecycle endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ReadinessCheck(BaseModel):
    """One named readiness check for a supporting model lifecycle state."""

    name: str = Field(description="Identifier for the readiness check.")
    passed: bool = Field(description="Whether this readiness check is satisfied.")
    detail: str | None = Field(
        default=None,
        description="Optional human-readable explanation for the check result.",
    )


class SupportingModelLifecycleResponse(BaseModel):
    """Lifecycle state snapshot for a supporting model."""

    model_id: str
    model_type: str = Field(
        description="Supporting model category (supporting_ml or supporting_rl)."
    )
    state: str = Field(description="Current registry lifecycle state.")
    model_path: str | None = Field(
        default=None,
        description="Persisted artifact path when an artifact has been loaded.",
    )
    trainer_class: str = Field(
        description="Fully-qualified trainer class backing the model."
    )
    input_data_types: list[str] = Field(
        default_factory=list,
        description="Configured input data categories for inference.",
    )
    signal_type: str = Field(description="Output signal category emitted by the model.")
    algorithm: str | None = Field(
        default=None,
        description="Configured algorithm/backend identifier when available.",
    )
    last_error: str | None = Field(
        default=None,
        description="Most recent lifecycle error message, when present.",
    )
    is_ready: bool = Field(
        description="True when the model is in the READY state for inference."
    )
    readiness_checks: list[ReadinessCheck] = Field(
        default_factory=list,
        description="Per-condition readiness checks contributing to readiness.",
    )


class ActivatePretrainedRequest(BaseModel):
    """Request to activate a pretrained supporting ML backend.

    Hyperparameters are optional overrides on top of the values stored with
    the model configuration. They are used to build the concrete backend
    configuration (for example a sentiment ``SentimentConfig``).
    """

    hyperparameters: dict[str, int | float | str | bool] = Field(
        default_factory=dict,
        description=(
            "Optional hyperparameter overrides merged over the stored model "
            "hyperparameters when constructing the pretrained backend."
        ),
    )


class LoadArtifactRequest(BaseModel):
    """Request to load and validate an existing supporting model artifact."""

    model_path: str = Field(
        min_length=1,
        description=(
            "Filesystem path to the trained artifact. For NewsSentimentTrainer "
            "this may be a directory or a news_sentiment.json file; for "
            "supporting RL it is the model .zip path."
        ),
    )


__all__ = [
    "ReadinessCheck",
    "SupportingModelLifecycleResponse",
    "ActivatePretrainedRequest",
    "LoadArtifactRequest",
]
