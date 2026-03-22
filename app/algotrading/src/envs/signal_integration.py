"""Signal integration utilities for trading environments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
from algotrading.src.models.inference import TimestampAlignmentService


@dataclass(frozen=True, slots=True)
class SignalConfig:
    """Configuration for a supporting-model signal feature.

    Args:
        model_id: Supporting model identifier used for alignment lookup.
        signal_key: Key name used in debug metadata.
        default_value: Fallback value used when aligned signal is missing.
        normalize: Whether to clip signal values to normalize_range.
        normalize_range: Allowed range for signal value clipping.
        include_confidence: Whether to append confidence as a separate feature.
        include_staleness: Whether to append stale-indicator as a feature.
    """

    model_id: str
    signal_key: str
    default_value: float
    normalize: bool = True
    normalize_range: tuple[float, float] = (-1.0, 1.0)
    include_confidence: bool = True
    include_staleness: bool = False

    def __post_init__(self) -> None:
        """Validate immutable signal configuration."""

        if not self.model_id.strip():
            raise ValueError("model_id must be non-empty")
        if not self.signal_key.strip():
            raise ValueError("signal_key must be non-empty")

        low, high = self.normalize_range
        if low >= high:
            raise ValueError("normalize_range must be an increasing tuple")


class SignalIntegration:
    """Build aligned supporting-model signal features for environment observations."""

    def __init__(
        self,
        alignment_service: TimestampAlignmentService,
        signal_configs: list[SignalConfig],
    ) -> None:
        """Initialize signal integration dependencies and feature config."""

        self._alignment_service = alignment_service
        self._configs = list(signal_configs)

    def get_observation_space_shape(self) -> int:
        """Return total number of signal-derived observation features."""

        count = 0
        for config in self._configs:
            count += 1
            if config.include_confidence:
                count += 1
            if config.include_staleness:
                count += 1
        return count

    def get_observation_space_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return low/high bounds arrays for all configured signal features."""

        lows: list[float] = []
        highs: list[float] = []

        for config in self._configs:
            low, high = config.normalize_range
            lows.append(float(low))
            highs.append(float(high))

            if config.include_confidence:
                lows.append(0.0)
                highs.append(1.0)

            if config.include_staleness:
                lows.append(0.0)
                highs.append(1.0)

        return np.asarray(lows, dtype=np.float32), np.asarray(highs, dtype=np.float32)

    def get_signal_features(
        self,
        timestamp: datetime,
    ) -> tuple[np.ndarray, dict[str, dict[str, Any]]]:
        """Get aligned signal features and debug metadata for a timestamp.

        Args:
            timestamp: Query timestamp for alignment.

        Returns:
            Tuple of feature array and per-signal debug metadata.
        """

        query_timestamp = self._ensure_aware_timestamp(timestamp)

        aligned = self._alignment_service.align(
            query_timestamp,
            [config.model_id for config in self._configs],
        )

        features: list[float] = []
        debug_info: dict[str, dict[str, Any]] = {}

        for config in self._configs:
            aligned_signal = aligned.signals.get(config.model_id)

            if aligned_signal is not None and aligned_signal.signal is not None:
                value = float(aligned_signal.signal.value)
                confidence = float(aligned_signal.signal.confidence or 1.0)
                is_stale = bool(aligned_signal.is_stale)
                source_timestamp = aligned_signal.signal.timestamp
            else:
                value = float(config.default_value)
                confidence = 0.0
                is_stale = True
                source_timestamp = None

            if config.normalize:
                low, high = config.normalize_range
                value = float(np.clip(value, low, high))

            confidence = float(np.clip(confidence, 0.0, 1.0))

            features.append(value)
            if config.include_confidence:
                features.append(confidence)
            if config.include_staleness:
                features.append(1.0 if is_stale else 0.0)

            debug_info[config.signal_key] = {
                "model_id": config.model_id,
                "value": value,
                "confidence": confidence,
                "is_stale": is_stale,
                "timestamp": source_timestamp,
            }

        return np.asarray(features, dtype=np.float32), debug_info

    def _ensure_aware_timestamp(self, timestamp: datetime) -> datetime:
        """Convert input timestamp to timezone-aware datetime."""

        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            return timestamp.replace(tzinfo=timezone.utc)
        return timestamp
