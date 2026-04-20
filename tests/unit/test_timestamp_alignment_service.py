"""Unit tests for timestamp alignment service."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from algotrading.src.models.inference import (
    AlignmentConfig,
    SignalCache,
    TimestampAlignmentService,
)
from algotrading.src.models.signals import ModelSignal, SignalMetadata, SignalType


def _signal(model_id: str, value: float, timestamp: datetime) -> ModelSignal:
    return ModelSignal(
        timestamp=timestamp,
        signal_type=SignalType.SENTIMENT,
        value=value,
        confidence=0.7,
        symbol="AAPL",
        metadata=SignalMetadata(model_id=model_id, model_type="ml"),
    )


def test_align_single_timestamp_uses_latest_before() -> None:
    cache = SignalCache(max_age_seconds=600, max_signals_per_model=20)
    service = TimestampAlignmentService(signal_cache=cache)

    base = datetime.now(tz=UTC)
    cache.put("model-a", _signal("model-a", 1.0, base + timedelta(seconds=5)))
    cache.put("model-a", _signal("model-a", 2.0, base + timedelta(seconds=10)))
    cache.put("model-a", _signal("model-a", 3.0, base + timedelta(seconds=20)))

    service.configure_model(
        "model-a",
        AlignmentConfig(model_id="model-a", max_staleness_seconds=60),
    )

    aligned = service.align(timestamp=base + timedelta(seconds=12))

    assert aligned.missing_models == []
    assert aligned.has_stale is False
    assert aligned.signals["model-a"].signal is not None
    assert aligned.signals["model-a"].signal.value == 2.0


def test_align_staleness_detection_and_forward_fill_toggle(caplog) -> None:
    cache = SignalCache(max_age_seconds=600, max_signals_per_model=20)
    service = TimestampAlignmentService(signal_cache=cache)

    base = datetime.now(tz=UTC)
    cache.put("model-a", _signal("model-a", 1.0, base))

    service.configure_model(
        "model-a",
        AlignmentConfig(
            model_id="model-a",
            max_staleness_seconds=5,
            forward_fill=False,
            warn_on_stale=True,
        ),
    )

    aligned = service.align(timestamp=base + timedelta(seconds=12))
    record = aligned.signals["model-a"]

    assert aligned.has_stale is True
    assert record.is_stale is True
    assert record.signal is None
    assert record.staleness_seconds == 12.0
    assert "Stale signal for model-a" in caplog.text


def test_align_uses_default_value_when_signal_missing() -> None:
    cache = SignalCache(max_age_seconds=600, max_signals_per_model=20)
    service = TimestampAlignmentService(signal_cache=cache)

    base = datetime.now(tz=UTC)
    service.configure_model(
        "model-b",
        AlignmentConfig(
            model_id="model-b",
            max_staleness_seconds=60,
            default_value=-0.25,
        ),
    )

    aligned = service.align(timestamp=base)
    record = aligned.signals["model-b"]

    assert aligned.missing_models == []
    assert record.used_default is True
    assert record.signal is not None
    assert record.signal.value == -0.25
    assert record.signal.metadata.model_type == "default"


def test_align_batch_preserves_input_order() -> None:
    cache = SignalCache(max_age_seconds=600, max_signals_per_model=20)
    service = TimestampAlignmentService(signal_cache=cache)

    base = datetime.now(tz=UTC)
    cache.put("model-a", _signal("model-a", 1.0, base + timedelta(seconds=5)))
    cache.put("model-a", _signal("model-a", 2.0, base + timedelta(seconds=10)))

    service.configure_model(
        "model-a",
        AlignmentConfig(model_id="model-a", max_staleness_seconds=60),
    )

    query_times = [
        base + timedelta(seconds=10),
        base + timedelta(seconds=6),
        base + timedelta(seconds=12),
    ]

    batch = service.align_batch(query_times)

    assert [entry.timestamp for entry in batch] == query_times
    assert batch[0].signals["model-a"].signal is not None
    assert batch[0].signals["model-a"].signal.value == 2.0
    assert batch[1].signals["model-a"].signal is not None
    assert batch[1].signals["model-a"].signal.value == 1.0


def test_align_reports_missing_models_without_defaults() -> None:
    cache = SignalCache(max_age_seconds=600, max_signals_per_model=20)
    service = TimestampAlignmentService(signal_cache=cache)

    base = datetime.now(tz=UTC)
    service.configure_model(
        "model-missing",
        AlignmentConfig(model_id="model-missing", max_staleness_seconds=30),
    )

    aligned = service.align(base)

    assert "model-missing" in aligned.missing_models
    assert "model-missing" not in aligned.signals


def test_align_rejects_naive_timestamp() -> None:
    cache = SignalCache(max_age_seconds=600, max_signals_per_model=20)
    service = TimestampAlignmentService(signal_cache=cache)

    with pytest.raises(ValueError, match="timestamp must be timezone-aware"):
        service.align(datetime(2026, 3, 1, 12, 0))
