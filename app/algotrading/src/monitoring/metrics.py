"""Thread-safe in-process metrics collection and Prometheus export."""

from __future__ import annotations

import math
import re
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock, RLock
from typing import ClassVar

_METRIC_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9_:]")


@dataclass(frozen=True, order=True, slots=True)
class MetricKey:
    """A metric name plus a stable set of tag key/value pairs."""

    name: str
    tags: tuple[tuple[str, str], ...] = ()

    def display_name(self) -> str:
        """Return a human-readable metric key used by JSON export."""

        if not self.tags:
            return self.name
        tag_text = ",".join(f"{key}={value}" for key, value in self.tags)
        return f"{self.name}{{{tag_text}}}"


@dataclass(slots=True)
class MetricValue:
    """Aggregated observations for a metric series."""

    max_history: int = 1000
    count: int = 0
    total: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf
    last: float = 0.0
    samples: deque[float] = field(init=False)

    def __post_init__(self) -> None:
        """Initialize bounded sample storage."""

        self.samples = deque(maxlen=self.max_history)

    def record(self, value: float) -> None:
        """Record one finite metric observation."""

        if not math.isfinite(value):
            raise ValueError("metric value must be finite")
        self.count += 1
        self.total += value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
        self.last = value
        self.samples.append(value)

    @property
    def mean(self) -> float:
        """Return the arithmetic mean of recorded observations."""

        return self.total / self.count if self.count else 0.0

    @property
    def p95(self) -> float:
        """Return the p95 value from retained samples."""

        if not self.samples:
            return 0.0
        sorted_samples = sorted(self.samples)
        index = min(
            len(sorted_samples) - 1, max(0, math.ceil(len(sorted_samples) * 0.95) - 1)
        )
        return sorted_samples[index]

    def to_dict(self) -> dict[str, float | int | None]:
        """Serialize aggregate values for JSON responses."""

        return {
            "count": self.count,
            "sum": self.total,
            "min": self.minimum if self.count else None,
            "max": self.maximum if self.count else None,
            "mean": self.mean,
            "p95": self.p95,
            "last": self.last if self.count else None,
        }


class MetricsCollector:
    """Process-wide metrics collector for counters, gauges, and observations."""

    _instance: ClassVar["MetricsCollector | None"] = None
    _instance_lock: ClassVar[Lock] = Lock()

    def __new__(cls, max_history: int = 1000) -> "MetricsCollector":
        """Return the singleton collector instance."""

        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize(max_history=max_history)
        return cls._instance

    def _initialize(self, *, max_history: int) -> None:
        self._lock = RLock()
        self._max_history = max(1, int(max_history))
        self._metrics: dict[MetricKey, MetricValue] = {}
        self._counters: dict[MetricKey, int] = {}
        self._gauges: dict[MetricKey, float] = {}

    def record(
        self,
        name: str,
        value: float,
        tags: dict[str, object] | None = None,
    ) -> None:
        """Record an observation metric value."""

        key = self._make_key(name, tags)
        with self._lock:
            metric = self._metrics.setdefault(
                key,
                MetricValue(max_history=self._max_history),
            )
            metric.record(float(value))

    def increment(
        self,
        name: str,
        value: int = 1,
        tags: dict[str, object] | None = None,
    ) -> None:
        """Increment a counter by a non-negative integer value."""

        if value < 0:
            raise ValueError("counter increment must be non-negative")
        key = self._make_key(name, tags)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + int(value)

    def gauge(
        self,
        name: str,
        value: float,
        tags: dict[str, object] | None = None,
    ) -> None:
        """Set a gauge to a finite numeric value."""

        if not math.isfinite(value):
            raise ValueError("gauge value must be finite")
        key = self._make_key(name, tags)
        with self._lock:
            self._gauges[key] = float(value)

    def get_all(self) -> dict[str, dict[str, object]]:
        """Return all collected metrics in a JSON-friendly shape."""

        with self._lock:
            return {
                "metrics": {
                    key.display_name(): value.to_dict()
                    for key, value in sorted(self._metrics.items())
                },
                "counters": {
                    key.display_name(): value
                    for key, value in sorted(self._counters.items())
                },
                "gauges": {
                    key.display_name(): value
                    for key, value in sorted(self._gauges.items())
                },
            }

    def get_prometheus_format(self) -> str:
        """Export metrics in Prometheus text exposition format."""

        with self._lock:
            lines: list[str] = []
            emitted_types: set[str] = set()
            for key, value in sorted(self._counters.items()):
                self._append_prometheus_value(
                    lines, key, value, "counter", emitted_types
                )
            for key, value in sorted(self._gauges.items()):
                self._append_prometheus_value(lines, key, value, "gauge", emitted_types)
            for key, metric in sorted(self._metrics.items()):
                for suffix, value, metric_type in (
                    ("count", metric.count, "counter"),
                    ("sum", metric.total, "gauge"),
                    ("min", metric.minimum if metric.count else 0.0, "gauge"),
                    ("max", metric.maximum if metric.count else 0.0, "gauge"),
                    ("mean", metric.mean, "gauge"),
                    ("p95", metric.p95, "gauge"),
                    ("last", metric.last if metric.count else 0.0, "gauge"),
                ):
                    derived_key = MetricKey(name=f"{key.name}_{suffix}", tags=key.tags)
                    self._append_prometheus_value(
                        lines, derived_key, value, metric_type, emitted_types
                    )
        return "\n".join(lines) + ("\n" if lines else "# no metrics recorded\n")

    def reset(self) -> None:
        """Clear all collected metrics."""

        with self._lock:
            self._metrics.clear()
            self._counters.clear()
            self._gauges.clear()

    def _make_key(
        self,
        name: str,
        tags: dict[str, object] | None,
    ) -> MetricKey:
        normalized_name = _normalize_name(name)
        normalized_tags = tuple(
            sorted(
                (str(tag_key).strip(), str(tag_value))
                for tag_key, tag_value in (tags or {}).items()
                if str(tag_key).strip()
            )
        )
        return MetricKey(normalized_name, normalized_tags)

    def _append_prometheus_value(
        self,
        lines: list[str],
        key: MetricKey,
        value: float | int,
        metric_type: str,
        emitted_types: set[str],
    ) -> None:
        metric_name = _sanitize_prometheus_name(key.name)
        if metric_name not in emitted_types:
            lines.append(f"# TYPE {metric_name} {metric_type}")
            emitted_types.add(metric_name)
        lines.append(f"{metric_name}{_format_labels(key.tags)} {value}")


def record_latency(
    name: str,
    start_time: float,
    tags: dict[str, object] | None = None,
) -> float:
    """Record elapsed milliseconds since ``start_time`` and return the value."""

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    MetricsCollector().record(f"{name}_latency_ms", latency_ms, tags=tags)
    return latency_ms


def _normalize_name(name: str) -> str:
    normalized = name.strip()
    if not normalized:
        raise ValueError("metric name must be non-empty")
    return normalized


def _sanitize_prometheus_name(name: str) -> str:
    sanitized = _METRIC_NAME_PATTERN.sub("_", name)
    if sanitized[0].isdigit():
        return f"_{sanitized}"
    return sanitized


def _format_labels(tags: tuple[tuple[str, str], ...]) -> str:
    if not tags:
        return ""
    label_text = ",".join(
        f'{_sanitize_prometheus_name(key)}="{_escape_label_value(value)}"'
        for key, value in tags
    )
    return f"{{{label_text}}}"


def _escape_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


__all__ = [
    "MetricKey",
    "MetricValue",
    "MetricsCollector",
    "record_latency",
]
