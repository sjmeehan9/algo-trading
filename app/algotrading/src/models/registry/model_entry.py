"""Datatypes for supporting model registry entries."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Mapping

from algotrading.src.data_pipeline import DataFrequency, DataType
from algotrading.src.models.signals import SignalType
from algotrading.src.trainers import MLTrainer, RLTrainer


class ModelState(str, Enum):
    """Lifecycle states for registered supporting models."""

    REGISTERED = "registered"
    LOADING = "loading"
    LOADED = "loaded"
    READY = "ready"
    ERROR = "error"
    UNLOADED = "unloaded"


@dataclass(slots=True)
class ModelEntryConfig:
    """Configuration metadata for one supporting model registration.

    Args:
        model_id: Unique model identifier.
        model_type: Model family, either ``ml`` or ``rl``.
        signal_type: Output signal category emitted by this model.
        trainer_class: Fully-qualified trainer class path.
        model_path: Optional persisted model artifact path.
        config: Arbitrary model configuration dictionary.
        input_data_types: Supported input data categories.
        input_frequency: Expected input frequency.
        description: Optional human-readable description.
    """

    model_id: str
    model_type: str
    signal_type: SignalType
    trainer_class: str
    input_data_types: list[DataType]
    input_frequency: DataFrequency
    model_path: str | None = None
    config: dict[str, object] = field(default_factory=dict)
    description: str | None = None

    def __post_init__(self) -> None:
        """Validate core configuration invariants."""

        if not self.model_id.strip():
            raise ValueError("model_id must be non-empty")

        normalized_type = self.model_type.strip().lower()
        if normalized_type not in {"ml", "rl"}:
            raise ValueError("model_type must be either 'ml' or 'rl'")
        self.model_type = normalized_type

        if not self.trainer_class.strip():
            raise ValueError("trainer_class must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Serialize configuration to JSON-compatible dictionary."""

        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "signal_type": self.signal_type.value,
            "trainer_class": self.trainer_class,
            "model_path": self.model_path,
            "config": dict(self.config),
            "input_data_types": [value.value for value in self.input_data_types],
            "input_frequency": self.input_frequency.value,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ModelEntryConfig:
        """Deserialize configuration from dictionary payload."""

        config_raw = data.get("config", {})
        config_map = config_raw if isinstance(config_raw, Mapping) else {}

        input_types_raw = data.get("input_data_types", [])
        input_types_list = input_types_raw if isinstance(input_types_raw, list) else []

        return cls(
            model_id=str(data["model_id"]),
            model_type=str(data["model_type"]),
            signal_type=SignalType(str(data["signal_type"])),
            trainer_class=str(data["trainer_class"]),
            model_path=(
                str(data["model_path"]) if data.get("model_path") is not None else None
            ),
            config={str(key): value for key, value in config_map.items()},
            input_data_types=[DataType(str(item)) for item in input_types_list],
            input_frequency=DataFrequency(str(data["input_frequency"])),
            description=(
                str(data["description"])
                if data.get("description") is not None
                else None
            ),
        )


@dataclass(slots=True)
class ModelEntry:
    """Runtime model registry entry with lifecycle and inference metadata."""

    config: ModelEntryConfig
    state: ModelState = ModelState.REGISTERED
    trainer: MLTrainer | RLTrainer | None = None
    last_inference_time: datetime | None = None
    inference_count: int = 0
    error_message: str | None = None
    registered_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    loaded_at: datetime | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize entry metadata for persistence.

        Trainer instances are intentionally excluded.
        """

        return {
            "config": self.config.to_dict(),
            "state": self.state.value,
            "last_inference_time": (
                self.last_inference_time.isoformat()
                if self.last_inference_time is not None
                else None
            ),
            "inference_count": self.inference_count,
            "error_message": self.error_message,
            "registered_at": self.registered_at.isoformat(),
            "loaded_at": (
                self.loaded_at.isoformat() if self.loaded_at is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ModelEntry:
        """Deserialize entry metadata from persistence payload."""

        config_raw = data.get("config", {})
        if not isinstance(config_raw, Mapping):
            raise ValueError("entry config payload must be a mapping")

        inference_count_raw = data.get("inference_count", 0)
        inference_count = (
            int(inference_count_raw)
            if isinstance(inference_count_raw, int | float | str)
            else 0
        )

        return cls(
            config=ModelEntryConfig.from_dict(config_raw),
            state=ModelState(str(data["state"])),
            trainer=None,
            last_inference_time=(
                datetime.fromisoformat(str(data["last_inference_time"]))
                if data.get("last_inference_time") is not None
                else None
            ),
            inference_count=inference_count,
            error_message=(
                str(data["error_message"])
                if data.get("error_message") is not None
                else None
            ),
            registered_at=datetime.fromisoformat(str(data["registered_at"])),
            loaded_at=(
                datetime.fromisoformat(str(data["loaded_at"]))
                if data.get("loaded_at") is not None
                else None
            ),
        )


__all__ = ["ModelState", "ModelEntryConfig", "ModelEntry"]
