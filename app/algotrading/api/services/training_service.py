"""Training job orchestration and persistence service."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, RLock
from typing import Any
from uuid import uuid4

from algotrading.api.schemas.training import (
    TrainingJob,
    TrainingJobCreate,
    TrainingJobStatus,
    TrainingProgress,
    is_terminal_status,
)
from algotrading.api.services.model_service import (
    ModelNotFoundError,
    ModelService,
)
from algotrading.api.websocket.manager import WebSocketManager
from algotrading.api.workers.training_worker import (
    TrainingExecutionResult,
    TrainingProgressUpdate,
    TrainingWorker,
)
from algotrading.src.models.tracking import GenerationTracker
from fastapi import Request

logger = logging.getLogger(__name__)


class TrainingServiceError(Exception):
    """Base exception for training service operations."""


class TrainingJobNotFoundError(TrainingServiceError):
    """Raised when a requested job does not exist."""


class TrainingJobStateError(TrainingServiceError):
    """Raised when an operation is invalid for a job's current state."""


class TrainingService:
    """Manage the lifecycle of asynchronous training jobs."""

    DEFAULT_TIMESTEPS = 100_000

    def __init__(
        self,
        ws_manager: WebSocketManager,
        model_service: ModelService,
        generation_tracker: GenerationTracker,
        worker: TrainingWorker,
        jobs_path: str | Path | None = None,
    ) -> None:
        """Initialize the service with collaborators and persistence path.

        Args:
            ws_manager: WebSocket manager used for progress broadcasts.
            model_service: Service used to resolve model configurations.
            generation_tracker: Generation tracker for training history.
            worker: Background worker that executes queued jobs.
            jobs_path: Optional path used to persist job records.
        """

        self._ws_manager = ws_manager
        self._model_service = model_service
        self._generation_tracker = generation_tracker
        self._worker = worker
        self._jobs_path = Path(jobs_path) if jobs_path else None

        self._lock = RLock()
        self._jobs: dict[str, TrainingJob] = {}
        self._job_requests: dict[str, TrainingJobCreate] = {}
        self._cancellation_events: dict[str, Event] = {}
        self._generation_ids: dict[str, str] = {}
        self._queue: deque[str] = deque()
        self._queue_event: asyncio.Event = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

        self._worker.attach(self)
        self._load_jobs()

    @property
    def generation_tracker(self) -> GenerationTracker:
        """Return the generation tracker collaborator."""

        return self._generation_tracker

    @property
    def worker(self) -> TrainingWorker:
        """Return the background worker collaborator."""

        return self._worker

    async def start(self) -> None:
        """Start the worker loop and bind the running event loop."""

        self._loop = asyncio.get_running_loop()
        self._worker.start()

    async def stop(self) -> None:
        """Stop the worker loop and persist any final state."""

        await self._worker.stop()
        self._loop = None

    def notify_shutdown(self) -> None:
        """Wake the worker so it can observe a stop signal."""

        if self._queue_event is not None:
            self._queue_event.set()

    async def create_job(self, request: TrainingJobCreate) -> TrainingJob:
        """Create and enqueue a new training job."""

        try:
            model = self._model_service.get_model(request.model_id)
        except ModelNotFoundError as exc:
            raise TrainingServiceError(str(exc)) from exc

        timesteps_candidate: object
        if request.total_timesteps is not None:
            timesteps_candidate = request.total_timesteps
        elif "total_timesteps" in request.training_config:
            timesteps_candidate = request.training_config["total_timesteps"]
        else:
            timesteps_candidate = model.hyperparameters.get(
                "total_timesteps", self.DEFAULT_TIMESTEPS
            )

        try:
            total_timesteps = int(timesteps_candidate)
        except (TypeError, ValueError) as exc:
            raise TrainingServiceError(
                "total_timesteps must be a positive integer"
            ) from exc

        if total_timesteps < 1:
            raise TrainingServiceError("total_timesteps must be a positive integer")

        job_id = f"job-{uuid4().hex[:12]}"
        job = TrainingJob(
            job_id=job_id,
            model_id=model.model_id,
            status=TrainingJobStatus.QUEUED,
            created_at=datetime.now(tz=UTC),
            total_timesteps=total_timesteps,
            description=request.description,
            training_config=dict(request.training_config),
            data_config=dict(request.data_config),
        )

        with self._lock:
            self._jobs[job_id] = job
            self._job_requests[job_id] = request
            self._cancellation_events[job_id] = Event()
            self._queue.append(job_id)
            self._persist_jobs_locked()

        self._signal_queue()
        await self._broadcast_job(job)
        return job

    async def get_job(self, job_id: str) -> TrainingJob:
        """Return one job by ID."""

        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise TrainingJobNotFoundError(f"Training job '{job_id}' not found")
        return job

    async def list_jobs(
        self,
        status: TrainingJobStatus | None = None,
        model_id: str | None = None,
    ) -> list[TrainingJob]:
        """List jobs with optional status and model filters."""

        with self._lock:
            jobs = list(self._jobs.values())
        if status is not None:
            jobs = [job for job in jobs if job.status == status]
        if model_id is not None:
            jobs = [job for job in jobs if job.model_id == model_id]
        jobs.sort(key=lambda item: item.created_at, reverse=True)
        return jobs

    async def cancel_job(self, job_id: str) -> TrainingJob:
        """Cancel a queued or running job."""

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise TrainingJobNotFoundError(f"Training job '{job_id}' not found")

            if is_terminal_status(job.status):
                raise TrainingJobStateError(
                    f"Cannot cancel job in terminal state '{job.status.value}'"
                )

            event = self._cancellation_events.get(job_id)
            if event is not None:
                event.set()

            if job.status == TrainingJobStatus.QUEUED:
                try:
                    self._queue.remove(job_id)
                except ValueError:
                    pass
                cancelled = job.model_copy(
                    update={
                        "status": TrainingJobStatus.CANCELLED,
                        "completed_at": datetime.now(tz=UTC),
                    }
                )
                self._jobs[job_id] = cancelled
                self._persist_jobs_locked()
                job_to_broadcast = cancelled
            else:
                job_to_broadcast = job

        await self._broadcast_job(job_to_broadcast)
        return job_to_broadcast

    async def wait_for_next_job(self, stop_event: asyncio.Event) -> str | None:
        """Block until a job is available or the worker is asked to stop."""

        while True:
            with self._lock:
                if stop_event.is_set():
                    return None
                if self._queue:
                    return self._queue[0]

            self._queue_event.clear()
            stop_task = asyncio.create_task(stop_event.wait())
            queue_task = asyncio.create_task(self._queue_event.wait())
            done, pending = await asyncio.wait(
                {stop_task, queue_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            for task in pending:
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            if stop_event.is_set():
                return None

    def get_cancellation_event(self, job_id: str) -> Event:
        """Return the cancellation event associated with a job."""

        with self._lock:
            event = self._cancellation_events.get(job_id)
            if event is None:
                event = Event()
                self._cancellation_events[job_id] = event
        return event

    async def prepare_job_execution(
        self, job_id: str
    ) -> tuple[TrainingJob, TrainingJobCreate, Any] | None:
        """Mark a job as running, resolve its model, and pop from queue."""

        with self._lock:
            job = self._jobs.get(job_id)
            request = self._job_requests.get(job_id)
            if job is None or request is None:
                if self._queue and self._queue[0] == job_id:
                    self._queue.popleft()
                return None

            cancellation_event = self._cancellation_events.get(job_id)
            if cancellation_event is not None and cancellation_event.is_set():
                if self._queue and self._queue[0] == job_id:
                    self._queue.popleft()
                cancelled = job.model_copy(
                    update={
                        "status": TrainingJobStatus.CANCELLED,
                        "completed_at": datetime.now(tz=UTC),
                    }
                )
                self._jobs[job_id] = cancelled
                self._persist_jobs_locked()
                cancelled_to_broadcast: TrainingJob | None = cancelled
            else:
                cancelled_to_broadcast = None

        if cancelled_to_broadcast is not None:
            await self._broadcast_job(cancelled_to_broadcast)
            return None

        try:
            model = self._model_service.get_model(request.model_id)
        except ModelNotFoundError as exc:
            await self.mark_failed(job_id, f"Model not found: {exc}")
            return None

        try:
            generation = self._generation_tracker.start_generation(
                model_id=model.model_id,
                hyperparameters=dict(model.hyperparameters),
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            await self.mark_failed(job_id, f"Failed to start generation: {exc}")
            return None

        with self._lock:
            if self._queue and self._queue[0] == job_id:
                self._queue.popleft()

            running = job.model_copy(
                update={
                    "status": TrainingJobStatus.RUNNING,
                    "started_at": datetime.now(tz=UTC),
                    "generation_id": generation.generation_id,
                }
            )
            self._jobs[job_id] = running
            self._generation_ids[job_id] = generation.generation_id
            self._persist_jobs_locked()

        await self._broadcast_job(running)
        return running, request, model

    async def handle_progress(
        self, job_id: str, update: TrainingProgressUpdate
    ) -> None:
        """Apply a progress update to a job and broadcast it via WebSocket."""

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status != TrainingJobStatus.RUNNING:
                return

            total = update.total_timesteps or job.total_timesteps or 0
            current = max(0, int(update.current_timestep))
            if total > 0:
                percent = min(100.0, max(0.0, (current / total) * 100.0))
            else:
                percent = 0.0

            updated = job.model_copy(
                update={
                    "progress_percent": percent,
                    "current_timestep": current,
                    "total_timesteps": total,
                    "current_metrics": dict(update.current_metrics),
                }
            )
            self._jobs[job_id] = updated

        progress = TrainingProgress(
            job_id=job_id,
            model_id=updated.model_id,
            status=updated.status,
            progress_percent=updated.progress_percent,
            current_timestep=updated.current_timestep,
            total_timesteps=updated.total_timesteps,
            current_metrics=dict(updated.current_metrics),
        )
        await self._broadcast_progress(progress)

    async def mark_completed(
        self,
        job_id: str,
        result: TrainingExecutionResult,
        elapsed_seconds: float,
    ) -> None:
        """Mark a running job as completed and update generation history."""

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            generation_id = self._generation_ids.get(job_id) or job.generation_id

            updated = job.model_copy(
                update={
                    "status": TrainingJobStatus.COMPLETED,
                    "completed_at": datetime.now(tz=UTC),
                    "progress_percent": 100.0,
                    "current_timestep": result.timesteps_trained,
                    "total_timesteps": max(
                        result.timesteps_trained, job.total_timesteps
                    ),
                    "current_metrics": dict(result.final_metrics),
                    "generation_id": generation_id,
                }
            )
            self._jobs[job_id] = updated
            self._persist_jobs_locked()

        if generation_id is not None and result.training_metrics is not None:
            try:
                self._generation_tracker.complete_generation(
                    generation_id=generation_id,
                    training_metrics=result.training_metrics,
                    model_path=result.model_path or "",
                )
            except Exception:
                logger.exception("Failed to mark generation %s complete", generation_id)

            if result.evaluation_metrics is not None:
                try:
                    self._generation_tracker.add_evaluation(
                        generation_id=generation_id,
                        eval_metrics=result.evaluation_metrics,
                    )
                except Exception:
                    logger.exception(
                        "Failed to attach evaluation metrics to generation %s",
                        generation_id,
                    )

        logger.info(
            "training_job_completed job_id=%s elapsed_seconds=%.2f",
            job_id,
            elapsed_seconds,
        )
        await self._broadcast_job(updated)

    async def mark_failed(self, job_id: str, error_message: str) -> None:
        """Mark a job as failed with the provided error message."""

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            generation_id = self._generation_ids.get(job_id) or job.generation_id

            try:
                if self._queue and self._queue[0] == job_id:
                    self._queue.popleft()
            except IndexError:
                pass

            updated = job.model_copy(
                update={
                    "status": TrainingJobStatus.FAILED,
                    "completed_at": datetime.now(tz=UTC),
                    "error_message": error_message,
                }
            )
            self._jobs[job_id] = updated
            self._persist_jobs_locked()

        if generation_id is not None:
            try:
                self._generation_tracker.fail_generation(
                    generation_id=generation_id,
                    error=error_message,
                )
            except Exception:
                logger.exception("Failed to mark generation %s failed", generation_id)

        await self._broadcast_job(updated)

    async def mark_cancelled(self, job_id: str) -> None:
        """Mark a running job as cancelled."""

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            generation_id = self._generation_ids.get(job_id) or job.generation_id

            try:
                if self._queue and self._queue[0] == job_id:
                    self._queue.popleft()
            except IndexError:
                pass

            updated = job.model_copy(
                update={
                    "status": TrainingJobStatus.CANCELLED,
                    "completed_at": datetime.now(tz=UTC),
                }
            )
            self._jobs[job_id] = updated
            self._persist_jobs_locked()

        if generation_id is not None:
            try:
                self._generation_tracker.fail_generation(
                    generation_id=generation_id,
                    error="Cancelled by user",
                )
            except Exception:
                logger.exception(
                    "Failed to mark generation %s cancelled", generation_id
                )

        await self._broadcast_job(updated)

    def _signal_queue(self) -> None:
        """Wake the worker by setting the queue event in a loop-safe way."""

        loop = self._loop
        if loop is None or not loop.is_running():
            self._queue_event.set()
            return

        loop.call_soon_threadsafe(self._queue_event.set)

    async def _broadcast_job(self, job: TrainingJob) -> None:
        """Broadcast a job state change to all listeners."""

        payload = job.model_dump(mode="json")
        await self._ws_manager.broadcast_to_topic(f"training:{job.job_id}", payload)
        await self._ws_manager.broadcast_to_topic(f"training:{job.model_id}", payload)
        await self._ws_manager.broadcast_to_topic(
            f"training:model:{job.model_id}", payload
        )

    async def _broadcast_progress(self, progress: TrainingProgress) -> None:
        """Broadcast incremental progress updates to subscribed clients."""

        payload = progress.model_dump(mode="json")
        await self._ws_manager.broadcast_to_topic(
            f"training:{progress.job_id}", payload
        )
        await self._ws_manager.broadcast_to_topic(
            f"training:{progress.model_id}", payload
        )
        await self._ws_manager.broadcast_to_topic(
            f"training:model:{progress.model_id}", payload
        )

    def _persist_jobs_locked(self) -> None:
        """Persist all jobs to disk while holding the internal lock."""

        if self._jobs_path is None:
            return

        self._jobs_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "saved_at": datetime.now(tz=UTC).isoformat(),
            "jobs": [job.model_dump(mode="json") for job in self._jobs.values()],
        }
        self._jobs_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _load_jobs(self) -> None:
        """Load persisted jobs from disk if available."""

        if self._jobs_path is None or not self._jobs_path.exists():
            return

        try:
            raw = json.loads(self._jobs_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Could not parse training jobs file %s", self._jobs_path)
            return

        for entry in raw.get("jobs", []):
            try:
                job = TrainingJob.model_validate(entry)
            except Exception:
                logger.exception("Skipping invalid persisted training job entry")
                continue

            # Any non-terminal jobs from a previous process are marked failed
            # because their worker context is gone.
            if not is_terminal_status(job.status):
                job = job.model_copy(
                    update={
                        "status": TrainingJobStatus.FAILED,
                        "error_message": (
                            "Job interrupted by API restart and was not resumed"
                        ),
                        "completed_at": datetime.now(tz=UTC),
                    }
                )

            with self._lock:
                self._jobs[job.job_id] = job


def create_default_training_service(
    *,
    ws_manager: WebSocketManager,
    model_service: ModelService,
    project_root: Path | None = None,
    worker: TrainingWorker | None = None,
) -> TrainingService:
    """Build a TrainingService with filesystem-backed defaults."""

    from algotrading.api.workers.training_worker import DefaultTrainingExecutor

    root = project_root or Path(__file__).resolve().parents[4]
    data_dir = root / "data" / "api"
    jobs_path = data_dir / "training_jobs.json"
    models_dir = root / "data" / "models"

    if worker is None:
        executor = DefaultTrainingExecutor(models_dir=models_dir)
        worker = TrainingWorker(executor=executor)

    return TrainingService(
        ws_manager=ws_manager,
        model_service=model_service,
        generation_tracker=model_service.generation_tracker,
        worker=worker,
        jobs_path=jobs_path,
    )


def get_training_service(request: Request) -> TrainingService:
    """FastAPI dependency resolver for the shared TrainingService instance."""

    service = getattr(request.app.state, "training_service", None)
    if service is None:
        from algotrading.api.services.model_service import get_model_service

        model_service = get_model_service(request)
        ws_manager = request.app.state.ws_manager
        service = create_default_training_service(
            ws_manager=ws_manager, model_service=model_service
        )
        request.app.state.training_service = service

        # Start the worker if there is a running event loop available.
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None
        if loop is not None and loop.is_running():
            asyncio.ensure_future(service.start(), loop=loop)
    return service


__all__ = [
    "TrainingService",
    "TrainingServiceError",
    "TrainingJobNotFoundError",
    "TrainingJobStateError",
    "create_default_training_service",
    "get_training_service",
]
