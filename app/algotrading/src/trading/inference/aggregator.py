"""Supporting signal aggregation for real-time trading observations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite

from algotrading.src.models.inference import TimestampAlignmentService
from algotrading.src.models.signals import ModelSignal

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AlignedSignal:
    """Signal value aligned to a market-state timestamp.

    Args:
        signal_id: Source model or strategy identifier.
        value: Fresh signal value when available, otherwise the default value.
        original_timestamp: Original source signal timestamp, if one was found.
        aligned_timestamp: Market timestamp used as the alignment target.
        age_seconds: Signal age at the aligned timestamp, when known.
        is_stale: Whether the source signal is too old or missing.
        default_value: Neutral fallback used for stale or missing signals.
        confidence: Signal confidence, defaulting to ``0.0`` for fallbacks.
        used_default: Whether ``value`` came from ``default_value``.
    """

    signal_id: str
    value: float
    original_timestamp: datetime | None
    aligned_timestamp: datetime
    age_seconds: float | None
    is_stale: bool
    default_value: float
    confidence: float
    used_default: bool = False

    @property
    def value_for_observation(self) -> float:
        """Return the value that should be fed to the core model."""

        return self.default_value if self.is_stale else self.value


class SignalAggregator:
    """Combine cached and direct supporting model signals for core inference."""

    def __init__(self, alignment_service: TimestampAlignmentService) -> None:
        """Initialize aggregation around the Phase 4 alignment service."""

        self.alignment_service = alignment_service
        self._signal_defaults: dict[str, float] = {}

    def register_signal_default(self, signal_id: str, default_value: float) -> None:
        """Register a neutral fallback for a signal.

        Args:
            signal_id: Model or strategy identifier.
            default_value: Value used when the source signal is stale or missing.
        """

        normalized_id = self._normalize_signal_id(signal_id)
        if not isfinite(default_value):
            raise ValueError("default_value must be finite")
        self._signal_defaults[normalized_id] = float(default_value)

    def get_registered_signal_ids(self) -> list[str]:
        """Return signal identifiers with registered defaults."""

        return list(self._signal_defaults.keys())

    def aggregate(
        self,
        signals: dict[str, ModelSignal],
        target_timestamp: datetime,
        stale_threshold: float,
        expected_signal_ids: list[str] | None = None,
    ) -> dict[str, AlignedSignal]:
        """Aggregate signals at the requested market timestamp.

        Args:
            signals: Latest direct signals available from the supporting pipeline.
            target_timestamp: Market timestamp for the core RL observation.
            stale_threshold: Maximum signal age in seconds before fallback is used.
            expected_signal_ids: Optional model IDs that should be represented even
                when no current signal exists.

        Returns:
            Mapping from signal ID to aligned signal metadata.
        """

        if stale_threshold <= 0 or not isfinite(stale_threshold):
            raise ValueError("stale_threshold must be finite and greater than 0")

        timestamp = self._ensure_aware_timestamp(target_timestamp)
        signal_ids = self._resolve_signal_ids(signals, expected_signal_ids)
        if not signal_ids:
            return {}

        aligned_by_id = self._align_from_service(timestamp, signal_ids)
        output: dict[str, AlignedSignal] = {}

        for signal_id in signal_ids:
            signal = self._select_signal(
                signal_id=signal_id,
                direct_signals=signals,
                aligned_signals=aligned_by_id,
                target_timestamp=timestamp,
            )
            output[signal_id] = self._build_aligned_signal(
                signal_id=signal_id,
                signal=signal,
                target_timestamp=timestamp,
                stale_threshold=stale_threshold,
                forced_stale=aligned_by_id.get(signal_id, (None, False))[1],
            )

        return output

    def _resolve_signal_ids(
        self,
        signals: dict[str, ModelSignal],
        expected_signal_ids: list[str] | None,
    ) -> list[str]:
        """Build deterministic signal ID list for aggregation."""

        ids: set[str] = set(self._signal_defaults)
        ids.update(self.alignment_service.get_all_model_ids())
        ids.update(self._normalize_signal_id(key) for key in signals)
        if expected_signal_ids is not None:
            ids.update(self._normalize_signal_id(key) for key in expected_signal_ids)
        return sorted(ids)

    def _align_from_service(
        self,
        timestamp: datetime,
        signal_ids: list[str],
    ) -> dict[str, tuple[ModelSignal | None, bool]]:
        """Fetch alignment-service results for the target timestamp."""

        try:
            aligned = self.alignment_service.align(timestamp, signal_ids)
        except Exception:
            logger.exception("Signal alignment service failed")
            return {}

        resolved: dict[str, tuple[ModelSignal | None, bool]] = {}
        for signal_id, aligned_signal in aligned.signals.items():
            resolved[signal_id] = (aligned_signal.signal, bool(aligned_signal.is_stale))
        return resolved

    def _select_signal(
        self,
        signal_id: str,
        direct_signals: dict[str, ModelSignal],
        aligned_signals: dict[str, tuple[ModelSignal | None, bool]],
        target_timestamp: datetime,
    ) -> ModelSignal | None:
        """Choose the newest non-future signal from direct and aligned sources."""

        candidates: list[ModelSignal] = []
        direct = direct_signals.get(signal_id)
        if direct is not None and direct.timestamp <= target_timestamp:
            candidates.append(direct)

        aligned_signal = aligned_signals.get(signal_id, (None, False))[0]
        if aligned_signal is not None and aligned_signal.timestamp <= target_timestamp:
            candidates.append(aligned_signal)

        if not candidates:
            return None
        return max(candidates, key=lambda signal: signal.timestamp)

    def _build_aligned_signal(
        self,
        signal_id: str,
        signal: ModelSignal | None,
        target_timestamp: datetime,
        stale_threshold: float,
        forced_stale: bool,
    ) -> AlignedSignal:
        """Convert a source signal into observation-ready aligned metadata."""

        default_value = self._signal_defaults.get(signal_id, 0.0)
        if signal is None:
            return AlignedSignal(
                signal_id=signal_id,
                value=default_value,
                original_timestamp=None,
                aligned_timestamp=target_timestamp,
                age_seconds=None,
                is_stale=True,
                default_value=default_value,
                confidence=0.0,
                used_default=True,
            )

        age_seconds = (target_timestamp - signal.timestamp).total_seconds()
        is_stale = forced_stale or age_seconds > stale_threshold or age_seconds < 0
        return AlignedSignal(
            signal_id=signal_id,
            value=float(signal.value),
            original_timestamp=signal.timestamp,
            aligned_timestamp=target_timestamp,
            age_seconds=age_seconds,
            is_stale=is_stale,
            default_value=default_value,
            confidence=float(
                signal.confidence if signal.confidence is not None else 1.0
            ),
            used_default=is_stale,
        )

    def _normalize_signal_id(self, signal_id: str) -> str:
        """Normalize and validate a signal identifier."""

        normalized = signal_id.strip()
        if not normalized:
            raise ValueError("signal_id must be non-empty")
        return normalized

    def _ensure_aware_timestamp(self, timestamp: datetime) -> datetime:
        """Ensure a timestamp is timezone-aware for alignment lookups."""

        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            return timestamp.replace(tzinfo=UTC)
        return timestamp
