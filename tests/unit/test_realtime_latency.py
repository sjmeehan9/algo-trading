"""Unit tests for real-time latency tracking."""

from __future__ import annotations

from algotrading.src.trading.inference import LatencyTracker


def test_latency_tracker_records_context_measurement() -> None:
    """Context manager measurements should update last and stats values."""

    tracker = LatencyTracker(warning_threshold_ms=1000.0)

    with tracker.track("core_inference") as measurement:
        value = sum(range(5))

    assert value == 10
    assert measurement.elapsed_ms >= 0.0
    assert tracker.get_last("core_inference") == measurement.elapsed_ms
    stats = tracker.get_stats("core_inference")
    assert stats["count"] == 1
    assert stats["max_ms"] >= stats["min_ms"]


def test_latency_tracker_calculates_p95_and_trims_history() -> None:
    """Manual records should retain bounded history and deterministic p95."""

    tracker = LatencyTracker(warning_threshold_ms=1000.0, max_history=20)
    for value in range(1, 21):
        tracker.record("total_pipeline", float(value))

    stats = tracker.get_stats("total_pipeline")
    assert stats["count"] == 20
    assert stats["mean_ms"] == 10.5
    assert stats["p95_ms"] == 19.0
    assert stats["last_ms"] == 20.0

    trimmed = LatencyTracker(warning_threshold_ms=1000.0, max_history=3)
    for value in [1.0, 2.0, 3.0, 4.0]:
        trimmed.record("stage", value)

    trimmed_stats = trimmed.get_stats("stage")
    assert trimmed_stats["count"] == 3
    assert trimmed_stats["min_ms"] == 2.0
    assert trimmed_stats["max_ms"] == 4.0
