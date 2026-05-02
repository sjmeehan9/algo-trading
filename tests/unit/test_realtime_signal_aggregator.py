"""Unit tests for real-time signal aggregation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from algotrading.src.models.inference import SignalCache, TimestampAlignmentService
from algotrading.src.models.signals import ModelSignal, SignalMetadata, SignalType
from algotrading.src.trading.inference import SignalAggregator


def _signal(
    model_id: str,
    timestamp: datetime,
    value: float,
    confidence: float = 0.8,
) -> ModelSignal:
    """Build a deterministic test signal."""

    return ModelSignal(
        timestamp=timestamp,
        signal_type=SignalType.SENTIMENT,
        value=value,
        confidence=confidence,
        symbol="AAPL",
        metadata=SignalMetadata(model_id=model_id, model_type="ml"),
    )


def _aggregator(cache: SignalCache | None = None) -> SignalAggregator:
    """Build an aggregator with an alignment service."""

    return SignalAggregator(TimestampAlignmentService(cache or SignalCache()))


def test_aggregate_uses_fresh_direct_signal() -> None:
    """Fresh direct signals should flow into observation values."""

    target = datetime(2026, 1, 2, 14, 30, tzinfo=UTC)
    aggregator = _aggregator()
    aggregator.register_signal_default("sentiment", 0.0)

    aligned = aggregator.aggregate(
        signals={
            "sentiment": _signal("sentiment", target - timedelta(seconds=5), 0.45)
        },
        target_timestamp=target,
        stale_threshold=60.0,
    )

    sentiment = aligned["sentiment"]
    assert sentiment.value == 0.45
    assert sentiment.value_for_observation == 0.45
    assert sentiment.confidence == 0.8
    assert sentiment.is_stale is False
    assert sentiment.used_default is False


def test_aggregate_marks_stale_signal_and_uses_default_for_observation() -> None:
    """Signals older than the threshold should default for model input."""

    target = datetime(2026, 1, 2, 14, 30, tzinfo=UTC)
    aggregator = _aggregator()
    aggregator.register_signal_default("sentiment", -0.1)

    aligned = aggregator.aggregate(
        signals={
            "sentiment": _signal("sentiment", target - timedelta(seconds=120), 0.9)
        },
        target_timestamp=target,
        stale_threshold=60.0,
    )

    sentiment = aligned["sentiment"]
    assert sentiment.value == 0.9
    assert sentiment.value_for_observation == -0.1
    assert sentiment.age_seconds == 120.0
    assert sentiment.is_stale is True
    assert sentiment.used_default is True


def test_aggregate_represents_missing_expected_signal_with_default() -> None:
    """Expected but missing signals should be explicit stale defaults."""

    target = datetime(2026, 1, 2, 14, 30, tzinfo=UTC)
    aggregator = _aggregator()
    aggregator.register_signal_default("missing", 0.25)

    aligned = aggregator.aggregate(
        signals={},
        target_timestamp=target,
        stale_threshold=60.0,
        expected_signal_ids=["missing"],
    )

    missing = aligned["missing"]
    assert missing.value == 0.25
    assert missing.value_for_observation == 0.25
    assert missing.original_timestamp is None
    assert missing.age_seconds is None
    assert missing.is_stale is True
    assert missing.used_default is True


def test_aggregate_can_resolve_signal_from_alignment_cache() -> None:
    """Aggregator should query the Phase 4 alignment service cache."""

    target = datetime.now(tz=UTC)
    cache = SignalCache(max_age_seconds=600.0)
    cache.put("trend", _signal("trend", target - timedelta(seconds=10), 0.6))

    aggregator = _aggregator(cache)
    aggregator.register_signal_default("trend", 0.0)

    aligned = aggregator.aggregate(
        signals={},
        target_timestamp=target,
        stale_threshold=60.0,
        expected_signal_ids=["trend"],
    )

    trend = aligned["trend"]
    assert trend.value == 0.6
    assert trend.is_stale is False
    assert trend.used_default is False
