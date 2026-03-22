"""Timestamp alignment service for supporting model signals."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from algotrading.src.data_pipeline import DataType
from algotrading.src.data_pipeline.routing import MultiFrequencyBuffer
from algotrading.src.models.inference.cache import SignalCache
from algotrading.src.models.signals import ModelSignal, SignalMetadata, SignalType

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AlignmentConfig:
    """Configuration for per-model timestamp alignment behavior.

    Args:
        model_id: Supporting model identifier.
        max_staleness_seconds: Maximum allowed signal age before stale.
        default_value: Optional fallback value when no signal is available.
        forward_fill: Whether stale signals can still be returned.
        warn_on_stale: Whether stale alignments emit log warnings.
    """

    model_id: str
    max_staleness_seconds: float
    default_value: float | None = None
    forward_fill: bool = True
    warn_on_stale: bool = True

    def __post_init__(self) -> None:
        """Validate alignment configuration fields."""

        if not self.model_id.strip():
            raise ValueError("model_id must be non-empty")
        if self.max_staleness_seconds <= 0:
            raise ValueError("max_staleness_seconds must be greater than 0")


@dataclass(frozen=True, slots=True)
class AlignedSignal:
    """Aligned signal metadata for one model at a query timestamp."""

    model_id: str
    signal: ModelSignal | None
    is_stale: bool
    staleness_seconds: float | None = None
    used_default: bool = False


@dataclass(frozen=True, slots=True)
class AlignedSignals:
    """Alignment result for a single query timestamp."""

    timestamp: datetime
    signals: dict[str, AlignedSignal]
    has_stale: bool
    missing_models: list[str]


class TimestampAlignmentService:
    """Align supporting model outputs to core model timestamps."""

    def __init__(
        self,
        signal_cache: SignalCache,
        buffer: MultiFrequencyBuffer | None = None,
    ) -> None:
        """Initialize the alignment service.

        Args:
            signal_cache: Cache of recent model inference signals.
            buffer: Optional multi-frequency buffer for signal channel lookups.
        """

        self._cache = signal_cache
        self._buffer = buffer
        self._configs: dict[str, AlignmentConfig] = {}
        self._default_staleness_seconds = 300.0

    def configure_model(self, model_id: str, config: AlignmentConfig) -> None:
        """Register or replace alignment behavior for one model."""

        if model_id != config.model_id:
            raise ValueError("model_id must match config.model_id")
        self._configs[model_id] = config

    def configure_all(self, configs: list[AlignmentConfig]) -> None:
        """Register a list of per-model alignment configurations."""

        self._configs = {config.model_id: config for config in configs}

    def set_default_staleness(self, seconds: float) -> None:
        """Set fallback staleness threshold used for unconfigured models."""

        if seconds <= 0:
            raise ValueError("seconds must be greater than 0")
        self._default_staleness_seconds = seconds

    def align(
        self,
        timestamp: datetime,
        model_ids: list[str] | None = None,
    ) -> AlignedSignals:
        """Align signals to a single timestamp.

        Args:
            timestamp: Reference timestamp used for alignment.
            model_ids: Optional explicit model list. Defaults to configured models.

        Returns:
            Aligned signal bundle for the requested timestamp.
        """

        self._validate_timestamp(timestamp)

        target_ids = model_ids if model_ids is not None else list(self._configs.keys())
        result: dict[str, AlignedSignal] = {}
        missing: list[str] = []
        has_stale = False

        for model_id in target_ids:
            config = self._configs.get(
                model_id,
                AlignmentConfig(
                    model_id=model_id,
                    max_staleness_seconds=self._default_staleness_seconds,
                ),
            )

            signal = self._get_latest_signal(model_id=model_id, timestamp=timestamp)

            if signal is None:
                if config.default_value is not None:
                    result[model_id] = AlignedSignal(
                        model_id=model_id,
                        signal=self._create_default_signal(model_id, timestamp, config),
                        is_stale=False,
                        staleness_seconds=None,
                        used_default=True,
                    )
                else:
                    missing.append(model_id)
                continue

            age = (timestamp - signal.timestamp).total_seconds()
            is_stale = age > config.max_staleness_seconds
            has_stale = has_stale or is_stale

            if is_stale and config.warn_on_stale:
                logger.warning("Stale signal for %s: %.1fs old", model_id, age)

            result[model_id] = AlignedSignal(
                model_id=model_id,
                signal=signal if (config.forward_fill or not is_stale) else None,
                is_stale=is_stale,
                staleness_seconds=age,
                used_default=False,
            )

        return AlignedSignals(
            timestamp=timestamp,
            signals=result,
            has_stale=has_stale,
            missing_models=missing,
        )

    def align_batch(
        self,
        timestamps: list[datetime],
        model_ids: list[str] | None = None,
    ) -> list[AlignedSignals]:
        """Align signals for multiple timestamps while preserving input order."""

        sorted_timestamps = sorted(enumerate(timestamps), key=lambda item: item[1])
        ordered: list[AlignedSignals | None] = [None] * len(timestamps)

        for index, timestamp in sorted_timestamps:
            ordered[index] = self.align(timestamp=timestamp, model_ids=model_ids)

        return [entry for entry in ordered if entry is not None]

    def get_all_model_ids(self) -> list[str]:
        """Return configured model identifiers."""

        return list(self._configs.keys())

    def is_model_configured(self, model_id: str) -> bool:
        """Return whether a model has explicit alignment configuration."""

        return model_id in self._configs

    def _create_default_signal(
        self,
        model_id: str,
        timestamp: datetime,
        config: AlignmentConfig,
    ) -> ModelSignal:
        """Create a fallback signal when model output is missing."""

        if config.default_value is None:
            raise ValueError(
                "config.default_value must be set to create default signal"
            )

        return ModelSignal(
            timestamp=timestamp,
            signal_type=SignalType.CUSTOM,
            value=float(config.default_value),
            confidence=1.0,
            symbol=None,
            metadata=SignalMetadata(
                model_id=model_id,
                model_type="default",
                model_version="alignment-default",
            ),
        )

    def _get_latest_signal(
        self, model_id: str, timestamp: datetime
    ) -> ModelSignal | None:
        """Resolve latest signal from buffer first, then cache fallback."""

        buffered = self._get_signal_from_buffer(model_id=model_id, timestamp=timestamp)
        if buffered is not None:
            return buffered
        return self._cache.get_latest_before(model_id=model_id, timestamp=timestamp)

    def _get_signal_from_buffer(
        self,
        model_id: str,
        timestamp: datetime,
    ) -> ModelSignal | None:
        """Get aligned signal from optional multi-frequency buffer."""

        if self._buffer is None:
            return None

        try:
            aligned = self._buffer.get_aligned_state(
                timestamp=timestamp,
                channels=[DataType.SIGNAL],
            )
        except Exception:
            logger.exception("Failed to query signal channel from buffer")
            return None

        record = aligned.get(DataType.SIGNAL)
        if record is None:
            return None

        payload = record.payload

        direct_signal = payload.get("signal")
        payload_model_id = payload.get("model_id")
        if isinstance(direct_signal, ModelSignal) and payload_model_id == model_id:
            return direct_signal if direct_signal.timestamp <= timestamp else None

        signals_map = payload.get("signals")
        if isinstance(signals_map, dict):
            candidate = signals_map.get(model_id)
            if isinstance(candidate, ModelSignal):
                return candidate if candidate.timestamp <= timestamp else None

        if isinstance(direct_signal, ModelSignal):
            source_model = direct_signal.metadata.model_id
            if source_model == model_id:
                return direct_signal if direct_signal.timestamp <= timestamp else None

        return None

    def _validate_timestamp(self, timestamp: datetime) -> None:
        """Validate query timestamp is timezone-aware."""

        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
