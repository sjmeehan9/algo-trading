"""Asynchronous execution engine for supporting model inference tasks."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, cast

from algotrading.src.data_pipeline import DataBatch, DataRecord
from algotrading.src.models.registry.model_entry import ModelEntry
from algotrading.src.models.registry.strategy_registry import StrategyEntry
from algotrading.src.models.signals import ModelSignal, SignalMetadata

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class InferenceTask:
    """Submitted unit of inference work for one model and one input payload."""

    task_id: str
    model_id: str
    data: DataRecord | DataBatch
    submitted_at: datetime
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Completed inference result with timing and error status."""

    task_id: str
    model_id: str
    signal: ModelSignal | None
    success: bool
    error_message: str | None
    inference_time_ms: float


class _TrainerLike(Protocol):
    """Protocol for trainer-like objects used for prediction."""

    def predict(self, value: object) -> object:
        """Return one prediction or prediction payload."""


class InferenceExecutor:
    """Run supporting model inference concurrently in worker threads."""

    def __init__(self, max_workers: int = 4, default_timeout: float = 5.0) -> None:
        """Initialize worker pool and timeout defaults."""

        if max_workers <= 0:
            raise ValueError("max_workers must be greater than 0")
        if default_timeout <= 0:
            raise ValueError("default_timeout must be greater than 0")

        self._default_timeout = default_timeout
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="inference",
        )

    def submit(
        self,
        task: InferenceTask,
        model_entry: ModelEntry | StrategyEntry,
    ) -> Future[InferenceResult]:
        """Submit one inference task for asynchronous execution."""

        return self._executor.submit(self._execute_inference, task, model_entry)

    def submit_batch(
        self,
        tasks: list[InferenceTask],
        entries: list[ModelEntry | StrategyEntry],
    ) -> list[Future[InferenceResult]]:
        """Submit a batch of inference tasks and return futures."""

        if len(tasks) != len(entries):
            raise ValueError("tasks and entries must have matching lengths")

        return [self.submit(task, entry) for task, entry in zip(tasks, entries)]

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown worker pool and optionally wait for completion."""

        self._executor.shutdown(wait=wait, cancel_futures=not wait)

    def _execute_inference(
        self,
        task: InferenceTask,
        entry: ModelEntry | StrategyEntry,
    ) -> InferenceResult:
        """Execute one inference and convert output to a standardized signal."""

        start = time.perf_counter()
        timeout_seconds = (
            task.timeout_seconds if task.timeout_seconds > 0 else self._default_timeout
        )

        try:
            signal = self._run_with_timeout(
                timeout_seconds=timeout_seconds,
                callable_fn=lambda: self._compute_signal(task, entry),
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            return InferenceResult(
                task_id=task.task_id,
                model_id=task.model_id,
                signal=signal,
                success=True,
                error_message=None,
                inference_time_ms=elapsed_ms,
            )
        except TimeoutError:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return InferenceResult(
                task_id=task.task_id,
                model_id=task.model_id,
                signal=None,
                success=False,
                error_message=f"Inference timed out after {timeout_seconds:.2f}s",
                inference_time_ms=elapsed_ms,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception("Inference task failed for model %s", task.model_id)
            return InferenceResult(
                task_id=task.task_id,
                model_id=task.model_id,
                signal=None,
                success=False,
                error_message=str(exc),
                inference_time_ms=elapsed_ms,
            )

    def _run_with_timeout(
        self,
        timeout_seconds: float,
        callable_fn: Callable[[], ModelSignal],
    ) -> ModelSignal:
        """Run one callable with timeout and return resulting signal."""

        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference-call")
        future: Future[ModelSignal] | None = None
        try:
            future = pool.submit(callable_fn)
            return future.result(timeout=timeout_seconds)
        except TimeoutError:
            if future is not None:
                future.cancel()
            raise
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    def _compute_signal(
        self,
        task: InferenceTask,
        entry: ModelEntry | StrategyEntry,
    ) -> ModelSignal:
        """Compute one model signal from model or strategy provider."""

        if isinstance(entry, StrategyEntry):
            strategy_instance = entry.instance or entry.strategy_class()
            payload = (
                task.data.payload if isinstance(task.data, DataRecord) else task.data
            )
            signal = strategy_instance.predict(payload, task.data)
            if not isinstance(signal, ModelSignal):
                raise TypeError("Strategy predict() must return ModelSignal")
            return signal

        trainer = entry.trainer
        if trainer is None:
            raise RuntimeError(f"Model '{entry.config.model_id}' has no loaded trainer")

        data_payload: object
        data_symbol: str | None
        data_timestamp: datetime

        if isinstance(task.data, DataRecord):
            data_payload = task.data.payload
            data_symbol = task.data.symbol
            data_timestamp = task.data.timestamp
        else:
            data_payload = [record.payload for record in task.data.records]
            data_symbol = task.data.symbol
            data_timestamp = task.data.end_time

        prediction = self._predict(trainer, data_payload)
        return self._prediction_to_signal(
            prediction=prediction,
            entry=entry,
            symbol=data_symbol,
            timestamp=data_timestamp,
        )

    def _predict(self, trainer: _TrainerLike, payload: object) -> object:
        """Call trainer prediction using supported payload fallbacks."""

        try:
            return trainer.predict(payload)
        except TypeError:
            # Some trainers require a positional list/array-like wrapper.
            return trainer.predict([payload])

    def _prediction_to_signal(
        self,
        prediction: object,
        entry: ModelEntry,
        symbol: str | None,
        timestamp: datetime,
    ) -> ModelSignal:
        """Convert trainer output into the standardized signal shape."""

        if isinstance(prediction, ModelSignal):
            return prediction

        value: float
        confidence: float | None = None

        if hasattr(prediction, "value"):
            value = float(getattr(prediction, "value"))
            raw_confidence = getattr(prediction, "confidence", None)
            confidence = float(raw_confidence) if raw_confidence is not None else None
        elif isinstance(prediction, tuple) and prediction:
            value = float(prediction[0])
            if len(prediction) > 1 and prediction[1] is not None:
                confidence = float(prediction[1])
        elif isinstance(prediction, list) and prediction:
            first = prediction[0]
            if hasattr(first, "value"):
                value = self._coerce_float(getattr(first, "value"))
                raw_confidence = getattr(first, "confidence", None)
                confidence = (
                    float(raw_confidence) if raw_confidence is not None else None
                )
            else:
                value = self._coerce_float(first)
        else:
            value = self._coerce_float(prediction)

        metadata = SignalMetadata(
            model_id=entry.config.model_id,
            model_type=entry.config.model_type,
            inference_time_ms=None,
            extra={"source": "inference_executor"},
        )

        signal_timestamp = timestamp
        if signal_timestamp.tzinfo is None or signal_timestamp.utcoffset() is None:
            signal_timestamp = signal_timestamp.replace(tzinfo=UTC)

        return ModelSignal(
            timestamp=signal_timestamp,
            signal_type=entry.config.signal_type,
            value=value,
            confidence=confidence,
            symbol=symbol,
            metadata=metadata,
        )

    def _coerce_float(self, value: object) -> float:
        """Convert a prediction value to float with explicit type checks."""

        if isinstance(value, (int, float, str)):
            return float(value)
        if hasattr(value, "__float__"):
            return float(cast(Any, value))
        raise TypeError(f"Prediction value '{value}' is not float-convertible")
