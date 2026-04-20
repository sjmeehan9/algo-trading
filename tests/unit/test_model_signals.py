"""Unit tests for supporting model signal interfaces and validation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta

import pytest
from algotrading.src.models.signals import (
    ModelSignal,
    SignalMetadata,
    SignalType,
    SignalValidationError,
    raise_if_invalid,
    signal_from_json,
    signal_to_json,
    validate_signal,
    validate_signal_for_type,
)


def _metadata() -> SignalMetadata:
    return SignalMetadata(
        model_id="news-sentiment-v1",
        model_type="ml",
        model_version="1.0.0",
        inference_time_ms=12.4,
        input_data_hash="abc123",
        extra={"provider": "mock"},
    )


def _base_signal(
    signal_type: SignalType = SignalType.SENTIMENT,
    value: float = 0.75,
    confidence: float | None = 0.9,
) -> ModelSignal:
    return ModelSignal(
        timestamp=datetime.now(UTC) - timedelta(seconds=3),
        signal_type=signal_type,
        value=value,
        confidence=confidence,
        symbol="AAPL",
        metadata=_metadata(),
    )


def test_signal_type_enum_values() -> None:
    """SignalType exposes all required string values."""

    assert SignalType.SENTIMENT.value == "sentiment"
    assert SignalType.TREND.value == "trend"
    assert SignalType.VOLATILITY.value == "volatility"
    assert SignalType.INDICATOR.value == "indicator"
    assert SignalType.POSITION.value == "position"
    assert SignalType.CUSTOM.value == "custom"


def test_model_signal_creation_with_all_fields() -> None:
    """ModelSignal and metadata instantiate with expected field values."""

    signal = _base_signal()

    assert signal.signal_type is SignalType.SENTIMENT
    assert signal.value == 0.75
    assert signal.confidence == 0.9
    assert signal.symbol == "AAPL"
    assert signal.metadata.model_id == "news-sentiment-v1"
    assert signal.metadata.extra["provider"] == "mock"


def test_model_signal_is_immutable() -> None:
    """Frozen signal dataclasses reject runtime mutation."""

    signal = _base_signal()

    with pytest.raises(FrozenInstanceError):
        signal.value = 0.1  # type: ignore[misc]

    with pytest.raises(TypeError):
        signal.metadata.extra["provider"] = "other"  # type: ignore[index]


def test_validate_signal_success_for_valid_signal() -> None:
    """Valid signal passes both base and type-specific validation."""

    signal = _base_signal()

    is_valid, errors = validate_signal(signal)
    is_type_valid, type_errors = validate_signal_for_type(signal)

    assert is_valid is True
    assert errors == []
    assert is_type_valid is True
    assert type_errors == []
    assert signal.is_valid() is True


def test_validate_signal_rejects_future_timestamp_and_bad_confidence() -> None:
    """Validation catches future timestamps and confidence outside [0, 1]."""

    signal = ModelSignal(
        timestamp=datetime.now(UTC) + timedelta(seconds=30),
        signal_type=SignalType.TREND,
        value=0.2,
        confidence=1.2,
        symbol="MSFT",
        metadata=_metadata(),
    )

    is_valid, errors = validate_signal(signal)

    assert is_valid is False
    assert "timestamp cannot be in the future" in errors
    assert "confidence must be in range [0, 1]" in errors


@pytest.mark.parametrize(
    ("signal_type", "value", "expected_error"),
    [
        (SignalType.SENTIMENT, 1.2, "SENTIMENT signal value must be in range [-1, 1]"),
        (SignalType.POSITION, 3.0, "POSITION signal value must be one of {0, 1, 2}"),
        (
            SignalType.VOLATILITY,
            -0.1,
            "VOLATILITY signal value must be greater than or equal to 0",
        ),
    ],
)
def test_validate_signal_for_type_rejects_invalid_values(
    signal_type: SignalType,
    value: float,
    expected_error: str,
) -> None:
    """Type-level validation applies signal-specific constraints."""

    signal = _base_signal(signal_type=signal_type, value=value)

    is_valid, errors = validate_signal_for_type(signal)

    assert is_valid is False
    assert expected_error in errors


def test_json_serialization_round_trip() -> None:
    """Signal JSON serialization/deserialization preserves full payload equality."""

    signal = _base_signal(signal_type=SignalType.INDICATOR, value=42.0, confidence=None)

    json_payload = signal_to_json(signal)
    rebuilt = signal_from_json(json_payload)

    assert rebuilt == signal
    assert rebuilt.to_dict() == signal.to_dict()


def test_signal_age_seconds_calculation() -> None:
    """Signal age uses reference timestamps and returns expected seconds."""

    timestamp = datetime(2026, 3, 22, 10, 0, 0, tzinfo=UTC)
    now = datetime(2026, 3, 22, 10, 0, 7, tzinfo=UTC)
    signal = ModelSignal(
        timestamp=timestamp,
        signal_type=SignalType.CUSTOM,
        value=5.0,
        confidence=None,
        symbol=None,
        metadata=_metadata(),
    )

    assert signal.age_seconds(now=now) == 7.0


def test_raise_if_invalid_raises_signal_validation_error() -> None:
    """Error helper raises a domain-specific exception for invalid signals."""

    signal = _base_signal(signal_type=SignalType.POSITION, value=7.0)

    with pytest.raises(SignalValidationError):
        raise_if_invalid(signal)
