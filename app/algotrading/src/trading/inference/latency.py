"""Latency tracking utilities for the real-time trading pipeline."""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from math import ceil, isfinite
from threading import RLock
from typing import Iterator

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LatencyStats:
    """Summary statistics for one tracked pipeline stage."""

    stage: str
    count: int
    mean_ms: float
    min_ms: float
    max_ms: float
    p95_ms: float
    last_ms: float

    def to_dict(self) -> dict[str, float | int | str]:
        """Serialize latency statistics to a dictionary."""

        return {
            "stage": self.stage,
            "count": self.count,
            "mean_ms": self.mean_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "p95_ms": self.p95_ms,
            "last_ms": self.last_ms,
        }


@dataclass(slots=True)
class _LatencyMeasurement:
    """Mutable measurement yielded by ``LatencyTracker.track``."""

    stage: str
    elapsed_ms: float = 0.0


class LatencyTracker:
    """Record per-stage latency with bounded history and warning logs."""

    def __init__(
        self,
        warning_threshold_ms: float = 50.0,
        max_history: int = 1000,
    ) -> None:
        """Initialize the tracker.

        Args:
            warning_threshold_ms: Log warning threshold per measured stage.
            max_history: Maximum retained observations per stage.
        """

        if warning_threshold_ms <= 0 or not isfinite(warning_threshold_ms):
            raise ValueError("warning_threshold_ms must be finite and greater than 0")
        if max_history <= 0:
            raise ValueError("max_history must be greater than 0")

        self.warning_threshold_ms = float(warning_threshold_ms)
        self._max_history = int(max_history)
        self._timings: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self._max_history)
        )
        self._current: dict[str, float] = {}
        self._lock = RLock()

    @contextmanager
    def track(self, stage: str) -> Iterator[_LatencyMeasurement]:
        """Track elapsed time for a code block.

        Args:
            stage: Stage name used for metrics and logs.

        Yields:
            Mutable measurement whose ``elapsed_ms`` is set when the block exits.
        """

        normalized_stage = self._normalize_stage(stage)
        measurement = _LatencyMeasurement(stage=normalized_stage)
        start = time.perf_counter()
        try:
            yield measurement
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            measurement.elapsed_ms = elapsed_ms
            self.record(normalized_stage, elapsed_ms)

    def record(self, stage: str, elapsed_ms: float) -> None:
        """Record an externally measured latency value."""

        normalized_stage = self._normalize_stage(stage)
        if elapsed_ms < 0 or not isfinite(elapsed_ms):
            raise ValueError("elapsed_ms must be finite and non-negative")

        with self._lock:
            self._timings[normalized_stage].append(float(elapsed_ms))
            self._current[normalized_stage] = float(elapsed_ms)

        if elapsed_ms > self.warning_threshold_ms:
            logger.warning(
                "Slow pipeline stage '%s': %.2fms",
                normalized_stage,
                elapsed_ms,
            )

    def get_last(self, stage: str) -> float:
        """Return the last recorded latency for a stage, or ``0.0``."""

        normalized_stage = self._normalize_stage(stage)
        with self._lock:
            return self._current.get(normalized_stage, 0.0)

    def get_stats(self, stage: str | None = None) -> dict[str, object]:
        """Return latency statistics for one stage or all tracked stages."""

        with self._lock:
            if stage is not None:
                normalized_stage = self._normalize_stage(stage)
                return self._calculate_stats(
                    normalized_stage,
                    list(self._timings.get(normalized_stage, [])),
                ).to_dict()

            return {
                stage_name: self._calculate_stats(
                    stage_name,
                    list(values),
                ).to_dict()
                for stage_name, values in self._timings.items()
            }

    def reset(self, stage: str | None = None) -> None:
        """Clear all timings or one stage's timings."""

        with self._lock:
            if stage is None:
                self._timings.clear()
                self._current.clear()
                return
            normalized_stage = self._normalize_stage(stage)
            self._timings.pop(normalized_stage, None)
            self._current.pop(normalized_stage, None)

    def _calculate_stats(self, stage: str, timings: list[float]) -> LatencyStats:
        """Calculate summary statistics for a stage."""

        if not timings:
            return LatencyStats(
                stage=stage,
                count=0,
                mean_ms=0.0,
                min_ms=0.0,
                max_ms=0.0,
                p95_ms=0.0,
                last_ms=0.0,
            )

        sorted_timings = sorted(timings)
        p95_index = min(len(sorted_timings) - 1, max(0, ceil(0.95 * len(timings)) - 1))
        return LatencyStats(
            stage=stage,
            count=len(timings),
            mean_ms=sum(timings) / len(timings),
            min_ms=sorted_timings[0],
            max_ms=sorted_timings[-1],
            p95_ms=sorted_timings[p95_index],
            last_ms=timings[-1],
        )

    def _normalize_stage(self, stage: str) -> str:
        """Normalize and validate a stage name."""

        normalized = stage.strip()
        if not normalized:
            raise ValueError("stage must be non-empty")
        return normalized
