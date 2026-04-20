"""Integration tests for timestamp alignment service."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from algotrading.src.data_pipeline import DataFrequency, DataRecord, DataType
from algotrading.src.data_pipeline.routing import (
    BufferConfig,
    ChannelConfig,
    MultiFrequencyBuffer,
)
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
        confidence=0.8,
        symbol="AMD",
        metadata=SignalMetadata(model_id=model_id, model_type="ml"),
    )


def test_alignment_with_real_signal_cache_lookup() -> None:
    cache = SignalCache(max_age_seconds=600, max_signals_per_model=30)
    service = TimestampAlignmentService(signal_cache=cache)

    base = datetime.now(tz=UTC)
    cache.put("news-model", _signal("news-model", 0.2, base + timedelta(seconds=2)))
    cache.put("news-model", _signal("news-model", 0.6, base + timedelta(seconds=7)))

    service.configure_model(
        "news-model",
        AlignmentConfig(model_id="news-model", max_staleness_seconds=120),
    )

    aligned = service.align(base + timedelta(seconds=8))
    result = aligned.signals["news-model"]

    assert result.signal is not None
    assert result.signal.value == 0.6
    assert result.is_stale is False


def test_alignment_prefers_buffer_signal_channel_over_cache() -> None:
    cache = SignalCache(max_age_seconds=600, max_signals_per_model=30)
    base = datetime.now(tz=UTC)

    cache.put(
        "news-model",
        _signal("news-model", 0.1, base + timedelta(seconds=5)),
    )

    buffer = MultiFrequencyBuffer(
        BufferConfig(
            channels=[
                ChannelConfig(
                    data_type=DataType.SIGNAL,
                    frequency=DataFrequency.IRREGULAR,
                    max_size=100,
                )
            ]
        )
    )

    buffer_signal = _signal("news-model", 0.9, base + timedelta(seconds=6))
    buffer.put(
        DataRecord(
            timestamp=buffer_signal.timestamp,
            data_type=DataType.SIGNAL,
            symbol="AMD",
            payload={"model_id": "news-model", "signal": buffer_signal},
            source_id="integration",
            frequency=DataFrequency.IRREGULAR,
        )
    )

    service = TimestampAlignmentService(signal_cache=cache, buffer=buffer)
    service.configure_model(
        "news-model",
        AlignmentConfig(model_id="news-model", max_staleness_seconds=60),
    )

    aligned = service.align(base + timedelta(seconds=7))
    result = aligned.signals["news-model"]

    assert result.signal is not None
    assert result.signal.value == 0.9


def test_alignment_falls_back_to_cache_when_buffer_has_no_matching_signal() -> None:
    cache = SignalCache(max_age_seconds=600, max_signals_per_model=30)
    base = datetime.now(tz=UTC)

    cache_signal = _signal("trend-model", 1.1, base + timedelta(seconds=4))
    cache.put("trend-model", cache_signal)

    buffer = MultiFrequencyBuffer(
        BufferConfig(
            channels=[
                ChannelConfig(
                    data_type=DataType.SIGNAL,
                    frequency=DataFrequency.IRREGULAR,
                    max_size=100,
                )
            ]
        )
    )

    other_signal = _signal("other-model", -0.3, base + timedelta(seconds=5))
    buffer.put(
        DataRecord(
            timestamp=other_signal.timestamp,
            data_type=DataType.SIGNAL,
            symbol="AMD",
            payload={"model_id": "other-model", "signal": other_signal},
            source_id="integration",
            frequency=DataFrequency.IRREGULAR,
        )
    )

    service = TimestampAlignmentService(signal_cache=cache, buffer=buffer)
    service.configure_model(
        "trend-model",
        AlignmentConfig(model_id="trend-model", max_staleness_seconds=60),
    )

    aligned = service.align(base + timedelta(seconds=6))
    result = aligned.signals["trend-model"]

    assert result.signal is not None
    assert result.signal.value == 1.1
