"""Unit tests for the signal cache used by inference pipeline."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from algotrading.src.models.inference import SignalCache
from algotrading.src.models.signals import ModelSignal, SignalMetadata, SignalType


def _signal(
    *,
    model_id: str,
    value: float,
    timestamp: datetime,
) -> ModelSignal:
    return ModelSignal(
        timestamp=timestamp,
        signal_type=SignalType.SENTIMENT,
        value=value,
        confidence=0.8,
        symbol="AAPL",
        metadata=SignalMetadata(model_id=model_id, model_type="ml"),
    )


def test_put_and_get_latest_and_history() -> None:
    cache = SignalCache(max_age_seconds=600, max_signals_per_model=3)
    now = datetime.now(tz=UTC)

    cache.put("model-a", _signal(model_id="model-a", value=1.0, timestamp=now))
    cache.put(
        "model-a",
        _signal(model_id="model-a", value=2.0, timestamp=now + timedelta(seconds=1)),
    )

    latest = cache.get_latest("model-a")
    assert latest is not None
    assert latest.value == 2.0

    history = cache.get_history("model-a", limit=2)
    assert [entry.value for entry in history] == [2.0, 1.0]


def test_get_latest_before_returns_expected_signal() -> None:
    cache = SignalCache(max_age_seconds=600, max_signals_per_model=10)
    start = datetime.now(tz=UTC)

    cache.put("model-a", _signal(model_id="model-a", value=1.0, timestamp=start))
    cache.put(
        "model-a",
        _signal(
            model_id="model-a",
            value=2.0,
            timestamp=start + timedelta(seconds=10),
        ),
    )
    cache.put(
        "model-a",
        _signal(
            model_id="model-a",
            value=3.0,
            timestamp=start + timedelta(seconds=20),
        ),
    )

    before = cache.get_latest_before("model-a", start + timedelta(seconds=11))
    assert before is not None
    assert before.value == 2.0

    none_before = cache.get_latest_before("model-a", start - timedelta(seconds=1))
    assert none_before is None


def test_prune_old_and_get_all_latest() -> None:
    cache = SignalCache(max_age_seconds=5, max_signals_per_model=10)
    now = datetime.now(tz=UTC)

    cache.put(
        "model-a",
        _signal(model_id="model-a", value=1.0, timestamp=now - timedelta(seconds=10)),
    )
    cache.put(
        "model-a",
        _signal(model_id="model-a", value=2.0, timestamp=now),
    )
    cache.put("model-b", _signal(model_id="model-b", value=5.0, timestamp=now))

    cache.prune_old()

    latest_map = cache.get_all_latest()
    assert set(latest_map.keys()) == {"model-a", "model-b"}
    assert latest_map["model-a"].value == 2.0
    assert latest_map["model-b"].value == 5.0
