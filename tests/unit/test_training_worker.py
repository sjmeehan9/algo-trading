"""Unit tests for training worker progress and cancellation handling."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from threading import Event
from typing import Any

import pytest
from algotrading.api.schemas.models import ModelConfigResponse, ModelType
from algotrading.api.schemas.training import (
    TrainingJob,
    TrainingJobCreate,
    TrainingJobStatus,
)
from algotrading.api.workers.training_worker import (
    DefaultTrainingExecutor,
    TrainingExecutionResult,
    TrainingExecutorConfigurationError,
    TrainingJobContext,
    TrainingProgressUpdate,
    TrainingWorker,
)


def _build_model() -> ModelConfigResponse:
    """Construct a minimal core RL model response for tests."""

    return ModelConfigResponse(
        model_id="model-test",
        name="Test",
        description=None,
        model_type=ModelType.CORE_RL,
        signal_type=None,
        trainer_type="stable_baselines3",
        algorithm="ppo",
        hyperparameters={"learning_rate": 0.0003},
        training_data_config={},
        supporting_model_ids=[],
        strategy_ids=[],
        environment_config={},
        reward_function="profit_seeker",
        input_data_types=[],
        input_frequency=None,
        created_at=datetime.now(tz=UTC),
        updated_at=datetime.now(tz=UTC),
        state="configured",
    )


def _build_job(model_id: str) -> TrainingJob:
    """Construct a queued training job for tests."""

    return TrainingJob(
        job_id="job-test",
        model_id=model_id,
        status=TrainingJobStatus.QUEUED,
        created_at=datetime.now(tz=UTC),
        total_timesteps=100,
    )


def test_default_executor_requires_environment_factory(tmp_path: Any) -> None:
    """RL training without an environment factory raises a clear error."""

    executor = DefaultTrainingExecutor(models_dir=tmp_path)
    model = _build_model()
    job = _build_job(model.model_id)
    context = TrainingJobContext(
        job=job,
        request=TrainingJobCreate(model_id=model.model_id, total_timesteps=10),
        model=model,
        cancellation_event=Event(),
        progress_callback=lambda update: None,
        generation_tracker=None,  # type: ignore[arg-type]
    )

    with pytest.raises(TrainingExecutorConfigurationError) as exc_info:
        executor.execute(context)

    assert "environment_factory" in str(exc_info.value)


def test_default_executor_requires_dataset_factory(tmp_path: Any) -> None:
    """ML training without a dataset factory raises a clear error."""

    executor = DefaultTrainingExecutor(models_dir=tmp_path)
    model = _build_model().model_copy(
        update={
            "model_type": ModelType.SUPPORTING_ML,
            "trainer_type": "sklearn",
            "algorithm": "random_forest",
            "input_data_types": ["news_text"],
            "input_frequency": "1m",
        }
    )
    job = _build_job(model.model_id)
    context = TrainingJobContext(
        job=job,
        request=TrainingJobCreate(model_id=model.model_id, total_timesteps=10),
        model=model,
        cancellation_event=Event(),
        progress_callback=lambda update: None,
        generation_tracker=None,  # type: ignore[arg-type]
    )

    with pytest.raises(TrainingExecutorConfigurationError) as exc_info:
        executor.execute(context)

    assert "dataset_factory" in str(exc_info.value)


class _FakeService:
    """Minimal fake service exposing the worker collaboration interface."""

    def __init__(self) -> None:
        self.queue: list[str] = ["job-1"]
        self.queue_event = asyncio.Event()
        self.calls: list[str] = []
        self.completed_results: list[TrainingExecutionResult] = []
        self.progress_updates: list[TrainingProgressUpdate] = []
        self.cancellation_event = Event()
        self.failed_messages: list[str] = []
        self.cancelled_jobs: list[str] = []
        self.shutdown = False
        self.generation_tracker = None  # not used by the fake executor below
        self._delivered_initial_job = False

    async def wait_for_next_job(self, stop_event: asyncio.Event) -> str | None:
        if not self.queue or stop_event.is_set():
            return None
        return self.queue[0]

    def get_cancellation_event(self, job_id: str) -> Event:
        self.calls.append(f"cancel_event:{job_id}")
        return self.cancellation_event

    async def prepare_job_execution(
        self, job_id: str
    ) -> tuple[TrainingJob, TrainingJobCreate, ModelConfigResponse] | None:
        self.calls.append(f"prepare:{job_id}")
        self.queue.remove(job_id)
        model = _build_model()
        job = _build_job(model.model_id).model_copy(
            update={"job_id": job_id, "status": TrainingJobStatus.RUNNING}
        )
        request = TrainingJobCreate(model_id=model.model_id, total_timesteps=10)
        return job, request, model

    async def handle_progress(
        self, job_id: str, update: TrainingProgressUpdate
    ) -> None:
        self.progress_updates.append(update)

    async def mark_completed(
        self,
        job_id: str,
        result: TrainingExecutionResult,
        elapsed_seconds: float,
    ) -> None:
        self.calls.append(f"complete:{job_id}")
        self.completed_results.append(result)

    async def mark_failed(self, job_id: str, error_message: str) -> None:
        self.calls.append(f"failed:{job_id}")
        self.failed_messages.append(error_message)

    async def mark_cancelled(self, job_id: str) -> None:
        self.calls.append(f"cancelled:{job_id}")
        self.cancelled_jobs.append(job_id)

    def notify_shutdown(self) -> None:
        self.shutdown = True
        self.queue.clear()
        self.queue_event.set()


async def test_worker_invokes_executor_and_marks_complete() -> None:
    """The worker pulls a job from the service and reports completion."""

    class _Executor:
        def __init__(self) -> None:
            self.invocations: list[TrainingJobContext] = []

        def execute(self, context: TrainingJobContext) -> TrainingExecutionResult:
            self.invocations.append(context)
            context.progress_callback(
                TrainingProgressUpdate(
                    current_timestep=10,
                    total_timesteps=10,
                    current_metrics={"reward": 1.0},
                )
            )
            return TrainingExecutionResult(
                timesteps_trained=10,
                final_metrics={"final_reward": 1.0},
                training_metrics=None,
            )

    executor = _Executor()
    service = _FakeService()
    worker = TrainingWorker(executor=executor)
    worker.attach(service)
    worker.start()

    # Wait until the executor has been invoked.
    for _ in range(200):
        if service.completed_results:
            break
        await asyncio.sleep(0.01)

    await worker.stop()

    assert executor.invocations, "executor should be invoked"
    assert any(call.startswith("complete:") for call in service.calls)
    # Progress propagated through the loop.
    assert service.progress_updates, "progress should propagate"


async def test_worker_marks_failed_on_executor_exception() -> None:
    """An executor exception causes the worker to call mark_failed."""

    class _ErrorExecutor:
        def execute(self, context: TrainingJobContext) -> TrainingExecutionResult:
            raise RuntimeError("explode")

    service = _FakeService()
    worker = TrainingWorker(executor=_ErrorExecutor())
    worker.attach(service)
    worker.start()

    for _ in range(200):
        if service.failed_messages:
            break
        await asyncio.sleep(0.01)

    await worker.stop()

    assert service.failed_messages == ["explode"]


def test_worker_start_requires_attach() -> None:
    """Calling start() without attach() raises a clear runtime error."""

    class _NoopExecutor:
        def execute(self, context: TrainingJobContext) -> TrainingExecutionResult:
            return TrainingExecutionResult(timesteps_trained=0)

    worker = TrainingWorker(executor=_NoopExecutor())

    async def _drive() -> None:
        worker.start()
        # Allow the loop to schedule the task, then await it.
        await asyncio.sleep(0)
        assert worker._task is not None
        with pytest.raises(RuntimeError):
            await worker._task

    asyncio.run(_drive())
