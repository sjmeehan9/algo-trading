"""Public signal interface for supporting models and strategies."""

from algotrading.src.models.signals.signal import (
    ModelSignal,
    SignalMetadata,
    SignalType,
    signal_from_json,
    signal_to_json,
)
from algotrading.src.models.signals.validation import (
    SignalValidationError,
    raise_if_invalid,
    validate_signal,
    validate_signal_for_type,
)

__all__ = [
    "ModelSignal",
    "SignalMetadata",
    "SignalType",
    "SignalValidationError",
    "raise_if_invalid",
    "signal_from_json",
    "signal_to_json",
    "validate_signal",
    "validate_signal_for_type",
]
