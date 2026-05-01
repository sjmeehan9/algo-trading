"""Unit tests for the asynchronous training service."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from algotrading.api.schemas.training import (
    TrainingJobCreate,
    TrainingJobStatus,
)
from algotrading.api.services.model_service import ModelService
from algotrading.api.services.training_service import (
    TrainingJobNotFoundError,
    TrainingJobStateError,
    TrainingService,
    TrainingServiceError,
)
from algotrading.api.websocket.manager import WebSocketManager
from algotrading.api.workers.training_worker import (
    TrainingCancelledError,
    TrainingExecutionResult,
    TrainingJobContext,
    TrainingProgressUpdate,
    TrainingWorker,
)
from algotrading.src.models.registry import (
    CustomStrategyRegistry,
    SupportingModelRegistry,
)
from algotrading.src.models.tracking import (
    GenerationTracker,
    JsonFileStorage,
    TrainingMetrics,
)


def _build_model_service(tmp_path: Path) -> ModelService:
    """Build an isolated model service for tests."""

    supporting_registry = SupportingModelRegistry()
    strategy_registry = CustomStrategyRegistry(strategy_dirs=[], auto_scan=False)
    generation_tracker = GenerationTracker(
        JsonFileStorage(str(tmp_path / "generations"))
    )
    return ModelService(
        supporting_registry=supporting_registry,
        strategy_registry=strategy_registry,
        generation_tracker=generation_tracker,
        core_models_path=tmp_path / "core_models.json",
    )


def _create_core_model(
    service: ModelService, total_timesteps: int | None = None
) -> str:
    """Create a baseline core RL model and return its ID."""

    from algotrading.api.schemas.models import ModelConfigCreate, ModelType

    hyperparameters: dict[str, float | int] = {
        "learning_rate": 0.0003,
        "n_steps": 2048,
    }
    if total_timesteps is not None:
        hyperparameters["total_timesteps"] = total_timesteps

    response = service.create_model(
        ModelConfigCreate(
            name="Test Core",
            model_type=ModelType.CORE_RL,
            trainer_type="stable_baselines3",
            algorithm="ppo",
            hyperparameters=hyperparameters,
            training_data_config={},
            supporting_model_ids=[],
            strategy_ids=[],
            environment_config={},
            reward_function="profit_seeker",
        )
    )
    return response.model_id


class _FakeExecutor:
    """Test executor recording invocations and producing deterministic output."""

    def __init__(
        self,
        *,
        progress_count: int = 2,
        cancel_during_progress: bool = False,
        raise_during_execute: Exception | None = None,
    ) -> None:
        self.progress_count = progress_count
        self.cancel_during_progress = cancel_during_progress
        self.raise_during_execute = raise_during_execute
        self.invocations: list[TrainingJobContext] = []

    def execute(self, context: TrainingJobContext) -> TrainingExecutionResult:
        self.invocations.append(context)
        if self.raise_during_execute is not None:
            raise self.raise_during_execute

        total = max(1, context.job.total_timesteps)
        for step in range(1, self.progress_count + 1):
            if self.cancel_during_progress and step == 1:
                # Simulate a worker noticing cancellation between progress events.
                while not context.cancellation_event.is_set():
                    pass
                raise TrainingCancelledError("cancelled mid-run")
            context.progress_callback(
                TrainingProgressUpdate(
                    current_timestep=int(total * (step / self.progress_count)),
                    total_timesteps=total,
                    current_metrics={"reward": float(step), "episodes": step},
                )
            )

        training_metrics = TrainingMetrics(
            final_reward=42.0,
            mean_reward=42.0,
            std_reward=0.0,
            episodes_completed=self.progress_count,
            timesteps_trained=total,
            training_time_seconds=0.01,
        )
        return TrainingExecutionResult(
            timesteps_trained=total,
            final_metrics={
                "final_reward": 42.0,
                "episodes_completed": self.progress_count,
            },
            training_metrics=training_metrics,
            model_path="/tmp/model.zip",
        )


@pytest.fixture()
def model_service(tmp_path: Path) -> ModelService:
    return _build_model_service(tmp_path)


@pytest.fixture()
def model_id(model_service: ModelService) -> str:
    return _create_core_model(model_service)


@pytest.mark.asyncio
async def test_create_job_validates_model_exists(
    tmp_path: Path, model_service: ModelService
) -> None:
    """Creating a job for an unknown model raises a service error."""

    executor = _FakeExecutor()
    worker = TrainingWorker(executor=executor, run_in_executor=_inline_executor)
    service = TrainingService(
        ws_manager=WebSocketManager(),
        model_service=model_service,
        generation_tracker=model_service.generation_tracker,
        worker=worker,
        jobs_path=tmp_path / "jobs.json",
    )

    with pytest.raises(TrainingServiceError):
        await service.create_job(TrainingJobCreate(model_id="nonexistent"))


@pytest.mark.asyncio
async def test_job_runs_to_completion_and_persists(
    tmp_path: Path, model_service: ModelService, model_id: str
) -> None:
    """A queued job is processed by the worker and reaches COMPLETED state."""

    executor = _FakeExecutor()
    worker = TrainingWorker(executor=executor, run_in_executor=_inline_executor)
    jobs_path = tmp_path / "jobs.json"
    service = TrainingService(
        ws_manager=WebSocketManager(),
        model_service=model_service,
        generation_tracker=model_service.generation_tracker,
        worker=worker,
        jobs_path=jobs_path,
    )

    await service.start()
    try:
        job = await service.create_job(
            TrainingJobCreate(model_id=model_id, total_timesteps=100)
        )
        assert job.status == TrainingJobStatus.QUEUED

        await _wait_for_status(service, job.job_id, TrainingJobStatus.COMPLETED)
    finally:
        await service.stop()

    completed = await service.get_job(job.job_id)
    assert completed.status == TrainingJobStatus.COMPLETED
    assert completed.progress_percent == 100.0
    assert completed.generation_id is not None
    assert completed.current_metrics["final_reward"] == 42.0
    assert jobs_path.exists()
    assert executor.invocations, "executor should have been invoked"


@pytest.mark.asyncio
async def test_create_job_uses_model_default_total_timesteps(
    tmp_path: Path, model_service: ModelService
) -> None:
    """Jobs should use model-level total_timesteps when request omits it."""

    model_with_default = _create_core_model(model_service, total_timesteps=321)

    executor = _FakeExecutor(progress_count=1)
    worker = TrainingWorker(executor=executor, run_in_executor=_inline_executor)
    service = TrainingService(
        ws_manager=WebSocketManager(),
        model_service=model_service,
        generation_tracker=model_service.generation_tracker,
        worker=worker,
        jobs_path=tmp_path / "jobs.json",
    )

    job = await service.create_job(TrainingJobCreate(model_id=model_with_default))
    assert job.total_timesteps == 321


@pytest.mark.asyncio
async def test_job_queue_is_fifo(
    tmp_path: Path, model_service: ModelService, model_id: str
) -> None:
    """Queued jobs should be executed in first-in-first-out order."""

    executor = _FakeExecutor(progress_count=1)
    worker = TrainingWorker(executor=executor, run_in_executor=_inline_executor)
    service = TrainingService(
        ws_manager=WebSocketManager(),
        model_service=model_service,
        generation_tracker=model_service.generation_tracker,
        worker=worker,
        jobs_path=tmp_path / "jobs.json",
    )

    first_job = await service.create_job(
        TrainingJobCreate(model_id=model_id, total_timesteps=10)
    )
    second_job = await service.create_job(
        TrainingJobCreate(model_id=model_id, total_timesteps=10)
    )

    await service.start()
    try:
        await _wait_for_status(service, first_job.job_id, TrainingJobStatus.COMPLETED)
        await _wait_for_status(service, second_job.job_id, TrainingJobStatus.COMPLETED)
    finally:
        await service.stop()

    invocation_order = [context.job.job_id for context in executor.invocations]
    assert invocation_order[:2] == [first_job.job_id, second_job.job_id]


@pytest.mark.asyncio
async def test_reorder_queue_changes_next_job(
    tmp_path: Path, model_service: ModelService, model_id: str
) -> None:
    """Queued jobs can be reordered before execution starts."""

    executor = _FakeExecutor(progress_count=1)
    worker = TrainingWorker(executor=executor, run_in_executor=_inline_executor)
    service = TrainingService(
        ws_manager=WebSocketManager(),
        model_service=model_service,
        generation_tracker=model_service.generation_tracker,
        worker=worker,
        jobs_path=tmp_path / "jobs.json",
    )

    first_job = await service.create_job(
        TrainingJobCreate(model_id=model_id, total_timesteps=10)
    )
    second_job = await service.create_job(
        TrainingJobCreate(model_id=model_id, total_timesteps=10)
    )

    reordered = await service.reorder_queue([second_job.job_id, first_job.job_id])
    next_job_id = await service.wait_for_next_job(asyncio.Event())

    assert [job.job_id for job in reordered] == [second_job.job_id, first_job.job_id]
    assert next_job_id == second_job.job_id


@pytest.mark.asyncio
async def test_job_failure_marks_status_failed(
    tmp_path: Path, model_service: ModelService, model_id: str
) -> None:
    """An executor exception transitions the job to FAILED with error_message."""

    executor = _FakeExecutor(raise_during_execute=RuntimeError("boom"))
    worker = TrainingWorker(executor=executor, run_in_executor=_inline_executor)
    service = TrainingService(
        ws_manager=WebSocketManager(),
        model_service=model_service,
        generation_tracker=model_service.generation_tracker,
        worker=worker,
        jobs_path=tmp_path / "jobs.json",
    )

    await service.start()
    try:
        job = await service.create_job(
            TrainingJobCreate(model_id=model_id, total_timesteps=100)
        )
        await _wait_for_status(service, job.job_id, TrainingJobStatus.FAILED)
    finally:
        await service.stop()

    failed = await service.get_job(job.job_id)
    assert failed.status == TrainingJobStatus.FAILED
    assert failed.error_message == "boom"


@pytest.mark.asyncio
async def test_cancel_queued_job_marks_cancelled_immediately(
    tmp_path: Path, model_service: ModelService, model_id: str
) -> None:
    """Cancelling a queued job should mark it cancelled and remove from queue."""

    # Use an executor that will block forever via cancellation_event.wait if reached.
    blocking_executor = _BlockingExecutor()
    worker = TrainingWorker(
        executor=blocking_executor, run_in_executor=_inline_executor
    )
    service = TrainingService(
        ws_manager=WebSocketManager(),
        model_service=model_service,
        generation_tracker=model_service.generation_tracker,
        worker=worker,
        jobs_path=tmp_path / "jobs.json",
    )

    # Pre-fill an in-flight job to keep queued items behind it.
    busy_job = await service.create_job(
        TrainingJobCreate(model_id=model_id, total_timesteps=10)
    )
    queued_job = await service.create_job(
        TrainingJobCreate(model_id=model_id, total_timesteps=10)
    )

    await service.start()
    try:
        # Wait until the first job has begun running.
        await _wait_for_status(service, busy_job.job_id, TrainingJobStatus.RUNNING)

        # Now cancel the still-queued second job.
        cancelled = await service.cancel_job(queued_job.job_id)
        assert cancelled.status == TrainingJobStatus.CANCELLED

        # Cancel the running one to allow worker to exit cleanly.
        await service.cancel_job(busy_job.job_id)
        await _wait_for_status(service, busy_job.job_id, TrainingJobStatus.CANCELLED)
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_cancel_terminal_job_raises_state_error(
    tmp_path: Path, model_service: ModelService, model_id: str
) -> None:
    """Cancelling a job in a terminal state raises a state error."""

    executor = _FakeExecutor()
    worker = TrainingWorker(executor=executor, run_in_executor=_inline_executor)
    service = TrainingService(
        ws_manager=WebSocketManager(),
        model_service=model_service,
        generation_tracker=model_service.generation_tracker,
        worker=worker,
        jobs_path=tmp_path / "jobs.json",
    )

    await service.start()
    try:
        job = await service.create_job(
            TrainingJobCreate(model_id=model_id, total_timesteps=10)
        )
        await _wait_for_status(service, job.job_id, TrainingJobStatus.COMPLETED)
        with pytest.raises(TrainingJobStateError):
            await service.cancel_job(job.job_id)
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_get_job_unknown_raises(
    tmp_path: Path, model_service: ModelService
) -> None:
    """Unknown job IDs raise TrainingJobNotFoundError."""

    executor = _FakeExecutor()
    worker = TrainingWorker(executor=executor, run_in_executor=_inline_executor)
    service = TrainingService(
        ws_manager=WebSocketManager(),
        model_service=model_service,
        generation_tracker=model_service.generation_tracker,
        worker=worker,
        jobs_path=tmp_path / "jobs.json",
    )

    with pytest.raises(TrainingJobNotFoundError):
        await service.get_job("missing-id")


@pytest.mark.asyncio
async def test_persisted_running_jobs_are_marked_failed_on_reload(
    tmp_path: Path, model_service: ModelService, model_id: str
) -> None:
    """Non-terminal jobs from a previous process should be marked failed."""

    jobs_path = tmp_path / "jobs.json"
    initial_payload = {
        "saved_at": datetime.now(tz=UTC).isoformat(),
        "jobs": [
            {
                "job_id": "job-orphan",
                "model_id": model_id,
                "status": "running",
                "created_at": datetime.now(tz=UTC).isoformat(),
                "total_timesteps": 100,
                "training_config": {},
                "data_config": {},
                "current_metrics": {},
                "progress_percent": 50.0,
                "current_timestep": 50,
            }
        ],
    }
    import json

    jobs_path.write_text(json.dumps(initial_payload), encoding="utf-8")

    executor = _FakeExecutor()
    worker = TrainingWorker(executor=executor, run_in_executor=_inline_executor)
    service = TrainingService(
        ws_manager=WebSocketManager(),
        model_service=model_service,
        generation_tracker=model_service.generation_tracker,
        worker=worker,
        jobs_path=jobs_path,
    )

    job = await service.get_job("job-orphan")
    assert job.status == TrainingJobStatus.FAILED
    assert job.error_message is not None


class _BlockingExecutor:
    """Executor that blocks on cancellation event to simulate long-running jobs."""

    def execute(self, context: TrainingJobContext) -> TrainingExecutionResult:
        # Block until cancellation is requested.
        context.progress_callback(
            TrainingProgressUpdate(
                current_timestep=1,
                total_timesteps=context.job.total_timesteps,
                current_metrics={"phase": "started"},
            )
        )
        context.cancellation_event.wait(timeout=10.0)
        if context.cancellation_event.is_set():
            raise TrainingCancelledError("cancelled")
        # Should not reach here under normal test paths.
        raise RuntimeError("blocking executor finished without cancellation")


async def _inline_executor(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Run blocking executor in a thread to retain real concurrency semantics."""

    return await asyncio.to_thread(func, *args, **kwargs)


async def _wait_for_status(
    service: TrainingService,
    job_id: str,
    status: TrainingJobStatus,
    timeout: float = 5.0,
) -> None:
    """Poll until the job reaches the expected status or timeout."""

    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        job = await service.get_job(job_id)
        if job.status == status:
            return
        await asyncio.sleep(0.05)
    job = await service.get_job(job_id)
    raise AssertionError(
        f"Job {job_id} did not reach status {status.value}; current status={job.status.value}"
    )
