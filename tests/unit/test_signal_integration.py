"""Unit tests for environment signal integration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
from algotrading.src.envs.signal_integration import SignalConfig, SignalIntegration
from algotrading.src.models.signals import ModelSignal, SignalMetadata, SignalType


class DummyAlignedSignal:
    """Minimal aligned signal object used by alignment test doubles."""

    def __init__(self, signal: ModelSignal | None, is_stale: bool) -> None:
        self.signal = signal
        self.is_stale = is_stale


class DummyAlignmentService:
    """Simple alignment service test double returning pre-wired results."""

    def __init__(self, results: dict[str, DummyAlignedSignal | None]) -> None:
        self._results = results
        self.last_timestamp: datetime | None = None
        self.last_model_ids: list[str] | None = None

    def align(self, timestamp: datetime, model_ids: list[str]) -> Any:
        self.last_timestamp = timestamp
        self.last_model_ids = list(model_ids)

        class _Aligned:
            def __init__(self, payload: dict[str, DummyAlignedSignal | None]) -> None:
                self.signals = payload

        return _Aligned(
            {model_id: self._results.get(model_id) for model_id in model_ids}
        )


def _signal(model_id: str, value: float, confidence: float = 0.75) -> ModelSignal:
    """Construct deterministic model signal for tests."""

    return ModelSignal(
        timestamp=datetime(2026, 1, 1, 14, 35, tzinfo=timezone.utc),
        signal_type=SignalType.SENTIMENT,
        value=value,
        confidence=confidence,
        symbol="AAPL",
        metadata=SignalMetadata(model_id=model_id, model_type="ml"),
    )


def test_observation_shape_and_bounds_include_optional_features() -> None:
    """Ensure shape and bounds account for confidence and staleness options."""

    service = DummyAlignmentService({})
    integration = SignalIntegration(
        alignment_service=service,  # type: ignore[arg-type]
        signal_configs=[
            SignalConfig(
                model_id="m1",
                signal_key="s1",
                default_value=0.0,
                include_confidence=True,
                include_staleness=True,
            ),
            SignalConfig(
                model_id="m2",
                signal_key="s2",
                default_value=0.0,
                include_confidence=False,
                include_staleness=False,
            ),
        ],
    )

    assert integration.get_observation_space_shape() == 4
    low, high = integration.get_observation_space_bounds()
    assert low.shape == (4,)
    assert high.shape == (4,)
    assert np.all(low <= high)


def test_get_signal_features_uses_default_for_missing_or_none() -> None:
    """Ensure missing aligned signals fall back to defaults and stale markers."""

    service = DummyAlignmentService({"news": None})
    integration = SignalIntegration(
        alignment_service=service,  # type: ignore[arg-type]
        signal_configs=[
            SignalConfig(
                model_id="news",
                signal_key="news_sentiment",
                default_value=-0.2,
                include_confidence=True,
                include_staleness=True,
            )
        ],
    )

    features, debug = integration.get_signal_features(
        datetime(2026, 1, 1, 14, 36, tzinfo=timezone.utc)
    )

    assert np.allclose(features, np.array([-0.2, 0.0, 1.0], dtype=np.float32))
    assert debug["news_sentiment"]["value"] == -0.2
    assert debug["news_sentiment"]["confidence"] == 0.0
    assert debug["news_sentiment"]["is_stale"] is True


def test_get_signal_features_clips_value_and_confidence() -> None:
    """Ensure configured clipping is applied to values and confidence."""

    aligned = DummyAlignedSignal(signal=_signal("news", 2.7, 2.0), is_stale=False)
    service = DummyAlignmentService({"news": aligned})
    integration = SignalIntegration(
        alignment_service=service,  # type: ignore[arg-type]
        signal_configs=[
            SignalConfig(
                model_id="news",
                signal_key="news_sentiment",
                default_value=0.0,
                normalize=True,
                normalize_range=(-1.0, 1.0),
                include_confidence=True,
                include_staleness=False,
            )
        ],
    )

    features, debug = integration.get_signal_features(
        datetime(2026, 1, 1, 14, 37, tzinfo=timezone.utc)
    )

    assert features.tolist() == [1.0, 1.0]
    assert debug["news_sentiment"]["value"] == 1.0
    assert debug["news_sentiment"]["confidence"] == 1.0


def test_get_signal_features_converts_naive_timestamp_to_aware() -> None:
    """Ensure alignment queries receive timezone-aware timestamps."""

    aligned = DummyAlignedSignal(signal=_signal("model", 0.3), is_stale=True)
    service = DummyAlignmentService({"model": aligned})
    integration = SignalIntegration(
        alignment_service=service,  # type: ignore[arg-type]
        signal_configs=[
            SignalConfig(
                model_id="model",
                signal_key="s",
                default_value=0.0,
                include_confidence=False,
                include_staleness=False,
            )
        ],
    )

    integration.get_signal_features(datetime(2026, 1, 1, 14, 40))

    assert service.last_timestamp is not None
    assert service.last_timestamp.tzinfo is not None
    assert service.last_model_ids == ["model"]
