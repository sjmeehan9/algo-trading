"""Standardized signal types produced by supporting models and strategies."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from types import MappingProxyType
from typing import Mapping


class SignalType(str, Enum):
    """Categories of model/strategy outputs consumed by the core RL model."""

    SENTIMENT = "sentiment"
    TREND = "trend"
    VOLATILITY = "volatility"
    INDICATOR = "indicator"
    POSITION = "position"
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class SignalMetadata:
    """Traceability metadata attached to a model signal.

    Args:
        model_id: Unique identifier of the source model or strategy.
        model_type: Source kind (for example ``ml``, ``rl``, or ``strategy``).
        model_version: Optional semantic/model generation version.
        inference_time_ms: Optional inference duration in milliseconds.
        input_data_hash: Optional hash/fingerprint of inference input data.
        extra: Additional provider-specific metadata.
    """

    model_id: str
    model_type: str
    model_version: str | None = None
    inference_time_ms: float | None = None
    input_data_hash: str | None = None
    extra: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate metadata invariants and freeze extra payload."""

        if not self.model_id.strip():
            raise ValueError("model_id must be non-empty")
        if not self.model_type.strip():
            raise ValueError("model_type must be non-empty")
        if self.inference_time_ms is not None and self.inference_time_ms < 0:
            raise ValueError("inference_time_ms must be greater than or equal to zero")

        # Preserve dataclass immutability expectations for nested metadata values.
        object.__setattr__(
            self,
            "extra",
            MappingProxyType(dict(self.extra)),
        )

    def to_dict(self) -> dict[str, object]:
        """Convert metadata to a JSON-compatible dictionary."""

        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "model_version": self.model_version,
            "inference_time_ms": self.inference_time_ms,
            "input_data_hash": self.input_data_hash,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> SignalMetadata:
        """Build metadata from a dictionary payload.

        Args:
            data: Mapping payload typically decoded from JSON.

        Returns:
            A `SignalMetadata` instance.
        """

        return cls(
            model_id=str(data.get("model_id", "")),
            model_type=str(data.get("model_type", "")),
            model_version=(
                str(data["model_version"])
                if data.get("model_version") is not None
                else None
            ),
            inference_time_ms=(
                float(data["inference_time_ms"])
                if data.get("inference_time_ms") is not None
                else None
            ),
            input_data_hash=(
                str(data["input_data_hash"])
                if data.get("input_data_hash") is not None
                else None
            ),
            extra=dict(data.get("extra", {})),
        )


@dataclass(frozen=True, slots=True)
class ModelSignal:
    """Standardized signal output produced by supporting models/strategies.

    Args:
        timestamp: Timestamp at which the signal was generated.
        signal_type: Semantic category of the signal value.
        value: Predicted signal value.
        confidence: Optional confidence score in range ``[0, 1]``.
        symbol: Optional instrument symbol associated with the signal.
        metadata: Source and traceability metadata.
    """

    timestamp: datetime
    signal_type: SignalType
    value: float
    metadata: SignalMetadata
    confidence: float | None = None
    symbol: str | None = None

    def __post_init__(self) -> None:
        """Validate intrinsic field constraints."""

        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        if not math.isfinite(self.value):
            raise ValueError("value must be finite")
        if self.confidence is not None and not math.isfinite(self.confidence):
            raise ValueError("confidence must be finite when provided")
        if self.symbol is not None and not self.symbol.strip():
            raise ValueError("symbol must be non-empty when provided")

    def to_dict(self) -> dict[str, object]:
        """Serialize the signal to a dictionary payload."""

        return {
            "timestamp": self.timestamp.isoformat(),
            "signal_type": self.signal_type.value,
            "value": self.value,
            "confidence": self.confidence,
            "symbol": self.symbol,
            "metadata": self.metadata.to_dict(),
        }

    def to_json(self) -> str:
        """Serialize the signal to a JSON string."""

        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ModelSignal:
        """Deserialize a signal from dictionary payload.

        Args:
            data: Mapping payload created by `to_dict`.

        Returns:
            A `ModelSignal` instance.
        """

        metadata_raw = data.get("metadata")
        metadata_payload: Mapping[str, object]
        if isinstance(metadata_raw, Mapping):
            metadata_payload = metadata_raw
        else:
            metadata_payload = {}

        return cls(
            timestamp=datetime.fromisoformat(str(data["timestamp"])),
            signal_type=SignalType(str(data["signal_type"])),
            value=float(data["value"]),
            confidence=(
                float(data["confidence"])
                if data.get("confidence") is not None
                else None
            ),
            symbol=str(data["symbol"]) if data.get("symbol") is not None else None,
            metadata=SignalMetadata.from_dict(metadata_payload),
        )

    @classmethod
    def from_json(cls, json_str: str) -> ModelSignal:
        """Deserialize a signal from its JSON representation."""

        return cls.from_dict(json.loads(json_str))

    def is_valid(self) -> bool:
        """Return `True` when the signal passes base and type-specific validation."""

        from algotrading.src.models.signals.validation import (
            validate_signal,
            validate_signal_for_type,
        )

        is_base_valid, _ = validate_signal(self)
        is_type_valid, _ = validate_signal_for_type(self)
        return is_base_valid and is_type_valid

    def age_seconds(self, now: datetime | None = None) -> float:
        """Return age in seconds for staleness checks.

        Args:
            now: Reference timestamp. Defaults to current UTC time.

        Returns:
            Age in fractional seconds.
        """

        reference = now or datetime.now(UTC)
        if reference.tzinfo is None or reference.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        return (reference - self.timestamp).total_seconds()


def signal_to_json(signal: ModelSignal) -> str:
    """Serialize a model signal to JSON."""

    return signal.to_json()


def signal_from_json(json_str: str) -> ModelSignal:
    """Deserialize a model signal from JSON."""

    return ModelSignal.from_json(json_str)
