"""Unit tests for asynchronous inference executor."""

from __future__ import annotations

import time
from datetime import UTC, datetime

from algotrading.src.data_pipeline import DataFrequency, DataRecord, DataType
from algotrading.src.models.inference import InferenceExecutor, InferenceTask
from algotrading.src.models.registry import ModelEntry, ModelEntryConfig
from algotrading.src.models.signals import ModelSignal, SignalMetadata, SignalType


class _FastTrainer:
    """Trainer that returns an immediate scalar prediction."""

    def predict(self, value: object) -> float:
        del value
        return 0.42


class _SlowTrainer:
    """Trainer that deliberately sleeps to trigger timeout behavior."""

    def __init__(self, delay_seconds: float) -> None:
        self._delay_seconds = delay_seconds

    def predict(self, value: object) -> float:
        del value
        time.sleep(self._delay_seconds)
        return 0.5


class _FailingTrainer:
    """Trainer that raises a runtime error during prediction."""

    def predict(self, value: object) -> float:
        del value
        raise RuntimeError("boom")


class _SignalTrainer:
    """Trainer that emits a fully-formed model signal."""

    def predict(self, value: object) -> ModelSignal:
        del value
        return ModelSignal(
            timestamp=datetime.now(tz=UTC),
            signal_type=SignalType.TREND,
            value=1.0,
            confidence=0.9,
            symbol="AAPL",
            metadata=SignalMetadata(model_id="signal-model", model_type="ml"),
        )


def _entry(model_id: str, trainer: object) -> ModelEntry:
    config = ModelEntryConfig(
        model_id=model_id,
        model_type="ml",
        signal_type=SignalType.SENTIMENT,
        trainer_class="tests.mocks.mock_ml_trainer.MockMLTrainer",
        input_data_types=[DataType.NEWS_TEXT],
        input_frequency=DataFrequency.IRREGULAR,
    )
    entry = ModelEntry(config=config)
    entry.trainer = trainer
    return entry


def _task(model_id: str, timeout: float = 0.5) -> InferenceTask:
    return InferenceTask(
        task_id=f"task-{model_id}",
        model_id=model_id,
        data=DataRecord(
            timestamp=datetime.now(tz=UTC),
            data_type=DataType.NEWS_TEXT,
            symbol="AAPL",
            payload={"headline": "Growth outlook improves"},
            frequency=DataFrequency.IRREGULAR,
        ),
        submitted_at=datetime.now(tz=UTC),
        timeout_seconds=timeout,
    )


def test_executor_success_with_scalar_prediction() -> None:
    executor = InferenceExecutor(max_workers=2, default_timeout=1.0)
    result = executor.submit(_task("fast"), _entry("fast", _FastTrainer())).result()
    executor.shutdown()

    assert result.success is True
    assert result.signal is not None
    assert result.signal.value == 0.42


def test_executor_timeout_handling() -> None:
    executor = InferenceExecutor(max_workers=2, default_timeout=0.05)
    result = executor.submit(
        _task("slow", timeout=0.05),
        _entry("slow", _SlowTrainer(delay_seconds=0.2)),
    ).result()
    executor.shutdown()

    assert result.success is False
    assert result.signal is None
    assert result.error_message is not None
    assert "timed out" in result.error_message.lower()


def test_executor_error_handling() -> None:
    executor = InferenceExecutor(max_workers=2, default_timeout=1.0)
    result = executor.submit(_task("bad"), _entry("bad", _FailingTrainer())).result()
    executor.shutdown()

    assert result.success is False
    assert result.signal is None
    assert result.error_message == "boom"


def test_executor_passthrough_model_signal() -> None:
    executor = InferenceExecutor(max_workers=2, default_timeout=1.0)
    result = executor.submit(
        _task("signal-model"),
        _entry("signal-model", _SignalTrainer()),
    ).result()
    executor.shutdown()

    assert result.success is True
    assert result.signal is not None
    assert result.signal.signal_type == SignalType.TREND
    assert result.signal.value == 1.0
