"""Thread-safe cache for latest supporting model signals."""

from __future__ import annotations

from collections import deque
from datetime import UTC, datetime, timedelta
from threading import RLock

from algotrading.src.models.signals import ModelSignal


class SignalCache:
    """Store and query recent model signals with bounded history and age pruning."""

    def __init__(
        self,
        max_age_seconds: float = 300,
        max_signals_per_model: int = 100,
    ) -> None:
        """Initialize an empty signal cache.

        Args:
            max_age_seconds: Maximum retained age for cached signals.
            max_signals_per_model: Maximum signal history length per model.
        """

        if max_age_seconds <= 0:
            raise ValueError("max_age_seconds must be greater than 0")
        if max_signals_per_model <= 0:
            raise ValueError("max_signals_per_model must be greater than 0")

        self._max_age = timedelta(seconds=max_age_seconds)
        self._max_signals_per_model = max_signals_per_model
        self._signals: dict[str, deque[ModelSignal]] = {}
        self._lock = RLock()

    def put(self, model_id: str, signal: ModelSignal) -> None:
        """Add a new signal to the model history."""

        if not model_id.strip():
            raise ValueError("model_id must be non-empty")

        with self._lock:
            queue = self._signals.setdefault(
                model_id,
                deque(maxlen=self._max_signals_per_model),
            )
            queue.append(signal)
            self._prune_old_locked(now=datetime.now(tz=UTC))

    def get_latest(self, model_id: str) -> ModelSignal | None:
        """Return the latest signal for a model if available."""

        with self._lock:
            queue = self._signals.get(model_id)
            if not queue:
                return None
            return queue[-1]

    def get_latest_before(
        self,
        model_id: str,
        timestamp: datetime,
    ) -> ModelSignal | None:
        """Return the newest signal at or before the given timestamp."""

        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")

        with self._lock:
            queue = self._signals.get(model_id)
            if not queue:
                return None
            for signal in reversed(queue):
                if signal.timestamp <= timestamp:
                    return signal
            return None

    def get_all_latest(self) -> dict[str, ModelSignal]:
        """Return the latest signal for every model currently cached."""

        with self._lock:
            latest: dict[str, ModelSignal] = {}
            for model_id, queue in self._signals.items():
                if queue:
                    latest[model_id] = queue[-1]
            return latest

    def get_history(self, model_id: str, limit: int = 10) -> list[ModelSignal]:
        """Return recent signal history for one model, newest first."""

        if limit <= 0:
            raise ValueError("limit must be greater than 0")

        with self._lock:
            queue = self._signals.get(model_id)
            if not queue:
                return []
            return list(reversed(list(queue)[-limit:]))

    def clear(self, model_id: str | None = None) -> None:
        """Clear all cached signals or those for one model."""

        with self._lock:
            if model_id is None:
                self._signals.clear()
                return
            self._signals.pop(model_id, None)

    def prune_old(self) -> int:
        """Prune signals older than configured max age and return removal count."""

        with self._lock:
            return self._prune_old_locked(now=datetime.now(tz=UTC))

    def _prune_old_locked(self, now: datetime) -> int:
        """Prune old signals under lock and return number removed."""

        cutoff = now - self._max_age
        removed = 0

        empty_models: list[str] = []
        for model_id, queue in self._signals.items():
            while queue and queue[0].timestamp < cutoff:
                queue.popleft()
                removed += 1
            if not queue:
                empty_models.append(model_id)

        for model_id in empty_models:
            self._signals.pop(model_id, None)

        return removed
