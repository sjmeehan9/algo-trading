"""Validation helpers for supporting model signals."""

from __future__ import annotations

import math
from datetime import UTC, datetime

from algotrading.src.models.signals.signal import ModelSignal, SignalType


class SignalValidationError(Exception):
    """Raised when a signal fails validation checks."""


def validate_signal(signal: ModelSignal) -> tuple[bool, list[str]]:
    """Validate common `ModelSignal` invariants.

    Args:
        signal: Signal instance to validate.

    Returns:
        Tuple of `(is_valid, errors)`.
    """

    errors: list[str] = []
    now = datetime.now(UTC)

    if signal.timestamp.tzinfo is None or signal.timestamp.utcoffset() is None:
        errors.append("timestamp must be timezone-aware")
    elif signal.timestamp > now:
        errors.append("timestamp cannot be in the future")

    if not math.isfinite(signal.value):
        errors.append("value must be finite")

    if signal.confidence is not None:
        if not math.isfinite(signal.confidence):
            errors.append("confidence must be finite when provided")
        elif signal.confidence < 0 or signal.confidence > 1:
            errors.append("confidence must be in range [0, 1]")

    if not signal.metadata.model_id.strip():
        errors.append("metadata.model_id must be non-empty")

    if not signal.metadata.model_type.strip():
        errors.append("metadata.model_type must be non-empty")

    return (len(errors) == 0, errors)


def validate_signal_for_type(signal: ModelSignal) -> tuple[bool, list[str]]:
    """Validate signal values against signal-type specific constraints.

    Args:
        signal: Signal instance to validate.

    Returns:
        Tuple of `(is_valid, errors)`.
    """

    errors: list[str] = []

    if signal.signal_type == SignalType.SENTIMENT:
        if signal.value < -1 or signal.value > 1:
            errors.append("SENTIMENT signal value must be in range [-1, 1]")

    if signal.signal_type == SignalType.POSITION:
        if signal.value not in {0.0, 1.0, 2.0}:
            errors.append("POSITION signal value must be one of {0, 1, 2}")

    if signal.signal_type == SignalType.VOLATILITY:
        if signal.value < 0:
            errors.append("VOLATILITY signal value must be greater than or equal to 0")

    return (len(errors) == 0, errors)


def raise_if_invalid(signal: ModelSignal) -> None:
    """Raise `SignalValidationError` when signal is invalid."""

    base_valid, base_errors = validate_signal(signal)
    type_valid, type_errors = validate_signal_for_type(signal)
    if base_valid and type_valid:
        return

    all_errors = base_errors + type_errors
    message = "; ".join(all_errors)
    raise SignalValidationError(message)
