"""Request and response schemas for model management endpoints."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from algotrading.src.models.signals import SignalType
from pydantic import BaseModel, Field, model_validator


class ModelType(str, Enum):
    """Supported model categories exposed by the API."""

    CORE_RL = "core_rl"
    SUPPORTING_ML = "supporting_ml"
    SUPPORTING_RL = "supporting_rl"


class ModelConfigBase(BaseModel):
    """Shared fields for all model configuration payloads."""

    name: str = Field(min_length=1, max_length=100)
    description: str | None = None
    model_type: ModelType
    signal_type: SignalType | None = None
    trainer_type: str = Field(min_length=1)
    algorithm: str = Field(min_length=1)


class CoreRLModelConfig(ModelConfigBase):
    """Configuration fields for core RL models."""

    model_type: ModelType = ModelType.CORE_RL
    hyperparameters: dict[str, int | float | str | bool] = Field(default_factory=dict)
    training_data_config: dict[str, object] = Field(default_factory=dict)
    supporting_model_ids: list[str] = Field(default_factory=list)
    strategy_ids: list[str] = Field(default_factory=list)
    environment_config: dict[str, object] = Field(default_factory=dict)
    reward_function: str = Field(min_length=1)


class SupportingModelConfig(ModelConfigBase):
    """Configuration fields for supporting ML/RL models."""

    hyperparameters: dict[str, int | float | str | bool] = Field(default_factory=dict)
    input_data_types: list[str] = Field(default_factory=list)
    input_frequency: str = Field(min_length=1)


class ModelConfigCreate(ModelConfigBase):
    """Create payload for model registrations."""

    hyperparameters: dict[str, int | float | str | bool] = Field(default_factory=dict)
    training_data_config: dict[str, object] = Field(default_factory=dict)
    supporting_model_ids: list[str] = Field(default_factory=list)
    strategy_ids: list[str] = Field(default_factory=list)
    environment_config: dict[str, object] = Field(default_factory=dict)
    reward_function: str | None = None
    input_data_types: list[str] = Field(default_factory=list)
    input_frequency: str | None = None

    @model_validator(mode="after")
    def validate_model_type_fields(self) -> ModelConfigCreate:
        """Validate required fields by model category."""

        if self.model_type == ModelType.CORE_RL:
            if not self.reward_function:
                raise ValueError("reward_function is required for core_rl models")
            return self

        if self.signal_type is None:
            raise ValueError("signal_type is required for supporting models")

        if not self.input_data_types:
            raise ValueError("input_data_types is required for supporting models")

        if not self.input_frequency:
            raise ValueError("input_frequency is required for supporting models")

        return self


class ModelConfigUpdate(BaseModel):
    """Partial update payload for model configurations."""

    name: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = None
    model_type: ModelType | None = None
    signal_type: SignalType | None = None
    trainer_type: str | None = Field(default=None, min_length=1)
    algorithm: str | None = Field(default=None, min_length=1)
    hyperparameters: dict[str, int | float | str | bool] | None = None
    training_data_config: dict[str, object] | None = None
    supporting_model_ids: list[str] | None = None
    strategy_ids: list[str] | None = None
    environment_config: dict[str, object] | None = None
    reward_function: str | None = None
    input_data_types: list[str] | None = None
    input_frequency: str | None = None


class ModelConfigResponse(ModelConfigCreate):
    """Model configuration returned by read/list endpoints."""

    model_id: str
    created_at: datetime
    updated_at: datetime
    state: str


__all__ = [
    "ModelType",
    "ModelConfigBase",
    "CoreRLModelConfig",
    "SupportingModelConfig",
    "ModelConfigCreate",
    "ModelConfigUpdate",
    "ModelConfigResponse",
]
