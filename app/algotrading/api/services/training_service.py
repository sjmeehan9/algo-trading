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

from algotrading.api.schemas.models import ModelType
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
        self._start_lock: asyncio.Lock | None = None
        self._start_lock_loop: asyncio.AbstractEventLoop | None = None

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

        await self.ensure_started()

    async def ensure_started(self) -> None:
        """Start the worker loop once for the active event loop."""

        loop = asyncio.get_running_loop()
        start_lock = self._get_start_lock(loop)
        async with start_lock:
            if self._worker.is_running:
                if self._loop is not None and self._loop is not loop:
                    raise TrainingServiceError(
                        "Training worker is already running on a different event loop"
                    )
                self._loop = loop
                return

            self._loop = loop
            self._worker.start()
            self._signal_queue()

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

        if request.continue_from_generation_id is not None:
            self._validate_continue_source(
                model=model,
                generation_id=request.continue_from_generation_id,
            )

        await self.ensure_started()

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
            continue_from_generation_id=request.continue_from_generation_id,
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

    def _validate_continue_source(self, *, model: Any, generation_id: str) -> None:
        """Validate a warm-start source generation before queueing the job.

        Raises:
            TrainingServiceError: If continue-training is requested for a
                non-RL model, the generation does not exist or belongs to a
                different model, or its saved artifact is unavailable.
        """

        if model.model_type not in (ModelType.CORE_RL, ModelType.SUPPORTING_RL):
            raise TrainingServiceError(
                "Continue training is only supported for RL models"
            )

        generation = self._generation_tracker.get_generation(generation_id)
        if generation is None:
            raise TrainingServiceError(
                f"Generation '{generation_id}' was not found"
            )
        if generation.model_id != model.model_id:
            raise TrainingServiceError(
                f"Generation '{generation_id}' belongs to a different model"
            )
        if not generation.model_path:
            raise TrainingServiceError(
                f"Generation '{generation_id}' has no saved model artifact to "
                "continue from"
            )
        if not _artifact_exists(generation.model_path):
            raise TrainingServiceError(
                f"Saved artifact for generation '{generation_id}' is missing: "
                f"{generation.model_path}"
            )

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
            queued_order = {job_id: index for index, job_id in enumerate(self._queue)}
        if status is not None:
            jobs = [job for job in jobs if job.status == status]
        if model_id is not None:
            jobs = [job for job in jobs if job.model_id == model_id]

        if status == TrainingJobStatus.QUEUED:
            jobs.sort(key=lambda item: queued_order.get(item.job_id, len(queued_order)))
        else:
            jobs.sort(key=lambda item: item.created_at, reverse=True)
        return jobs

    async def reorder_queue(self, job_ids: list[str]) -> list[TrainingJob]:
        """Replace the queued-job order and return the reordered queued jobs."""

        with self._lock:
            current_queue = list(self._queue)
            current_queue_set = set(current_queue)
            requested_queue_set = set(job_ids)

            missing_ids = requested_queue_set.difference(self._jobs)
            if missing_ids:
                missing = sorted(missing_ids)[0]
                raise TrainingJobNotFoundError(f"Training job '{missing}' not found")

            if requested_queue_set != current_queue_set:
                raise TrainingJobStateError(
                    "Queued job order must include exactly the currently queued jobs"
                )

            for job_id in job_ids:
                job = self._jobs[job_id]
                if job.status != TrainingJobStatus.QUEUED:
                    raise TrainingJobStateError(
                        f"Cannot reorder job '{job_id}' in state '{job.status.value}'"
                    )

            self._queue = deque(job_ids)
            self._persist_jobs_locked()
            reordered_jobs = [self._jobs[job_id] for job_id in job_ids]

        self._signal_queue()
        for job in reordered_jobs:
            await self._broadcast_job(job)
        return reordered_jobs

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
                parent_generation_id=request.continue_from_generation_id,
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

        promotion_error = await self._promote_supporting_model_if_needed(
            model_id=updated.model_id,
            result=result,
        )
        if promotion_error is not None:
            with self._lock:
                current = self._jobs.get(job_id)
                if current is not None:
                    updated = current.model_copy(
                        update={"error_message": promotion_error}
                    )
                    self._jobs[job_id] = updated
                    self._persist_jobs_locked()

        logger.info(
            "training_job_completed job_id=%s elapsed_seconds=%.2f",
            job_id,
            elapsed_seconds,
        )
        await self._broadcast_job(updated)

    async def _promote_supporting_model_if_needed(
        self,
        *,
        model_id: str,
        result: TrainingExecutionResult,
    ) -> str | None:
        """Promote a supporting model to READY after successful training.

        Supporting ML/RL training only completes the *training* step. The model
        is not usable for inference (and must not be selectable for core RL)
        until its freshly saved artifact can be loaded and exercised. This method
        detects supporting model types, hands the generation artifact to
        :class:`SupportingModelLifecycleService`, and only the lifecycle service's
        load/validate path transitions the registry entry to ``READY``.

        The training job itself remains ``COMPLETED`` regardless of promotion
        outcome (training did finish); a promotion failure is surfaced as the
        job's ``error_message`` and leaves the supporting registry entry in
        ``ERROR`` (or its prior non-ready state) with a clear message, so the
        model is not silently advertised as ready.

        Args:
            model_id: Identifier of the trained model.
            result: The successful training execution result, whose
                ``model_path`` points at the saved artifact.

        Returns:
            ``None`` when no promotion was required or promotion succeeded;
            otherwise a human-readable error message describing why the
            supporting model could not be made ready.
        """

        try:
            model = self._model_service.get_model(model_id)
        except ModelNotFoundError:
            return None

        if model.model_type not in (
            ModelType.SUPPORTING_ML,
            ModelType.SUPPORTING_RL,
        ):
            return None

        if not result.model_path:
            message = (
                "Supporting training completed but produced no artifact path; "
                "the model cannot be loaded for inference and was not marked "
                "ready"
            )
            self._mark_supporting_error(model_id, message)
            logger.error("supporting_promotion_no_artifact model_id=%s", model_id)
            return message

        # Imported lazily to avoid a circular import with the lifecycle service,
        # which depends on the model service that constructs this service.
        from algotrading.api.schemas.supporting_lifecycle import LoadArtifactRequest
        from algotrading.api.services.supporting_lifecycle_service import (
            SupportingLifecycleError,
            SupportingModelLifecycleService,
        )

        lifecycle_service = SupportingModelLifecycleService(
            model_service=self._model_service
        )

        try:
            await asyncio.to_thread(
                lifecycle_service.load_artifact,
                model_id,
                LoadArtifactRequest(model_path=result.model_path),
            )
        except SupportingLifecycleError as exc:
            message = (
                "Supporting training completed but the saved artifact could not "
                f"be loaded for inference: {exc}"
            )
            logger.error(
                "supporting_promotion_failed model_id=%s error=%s", model_id, exc
            )
            return message
        except Exception as exc:  # pragma: no cover - defensive guard
            message = (
                "Supporting training completed but readiness validation failed: "
                f"{exc}"
            )
            self._mark_supporting_error(model_id, str(exc))
            logger.exception(
                "supporting_promotion_unexpected_error model_id=%s", model_id
            )
            return message

        await self._broadcast_model_state(model_id)
        logger.info("supporting_promotion_ready model_id=%s", model_id)
        return None

    async def _broadcast_model_state(self, model_id: str) -> None:
        """Broadcast the refreshed model configuration after a state change.

        Emitting the updated model (including its new lifecycle ``state``) lets
        subscribed clients refresh the training selector and any readiness views
        without waiting for a manual reload.
        """

        try:
            model = self._model_service.get_model(model_id)
        except ModelNotFoundError:
            return

        payload = model.model_dump(mode="json")
        await self._ws_manager.broadcast_to_topic(f"models:{model_id}", payload)
        await self._ws_manager.broadcast_to_topic(f"training:model:{model_id}", payload)

    def _mark_supporting_error(self, model_id: str, message: str) -> None:
        """Best-effort transition of a supporting entry to ERROR state."""

        try:
            from algotrading.src.models.registry import ModelState

            self._model_service.supporting_registry.set_state(
                model_id, ModelState.ERROR, error=message
            )
        except Exception:  # pragma: no cover - defensive guard
            logger.debug(
                "Could not mark supporting model '%s' ERROR after promotion failure",
                model_id,
            )

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

    def _get_start_lock(self, loop: asyncio.AbstractEventLoop) -> asyncio.Lock:
        if self._start_lock is None or self._start_lock_loop is not loop:
            self._start_lock = asyncio.Lock()
            self._start_lock_loop = loop
        return self._start_lock

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
    api_config: Any | None = None,
    worker: TrainingWorker | None = None,
) -> TrainingService:
    """Build a TrainingService with filesystem-backed defaults."""

    from algotrading.api.config import APIConfig
    from algotrading.api.services.data_acquisition_service import (
        create_default_data_acquisition_service,
    )
    from algotrading.api.training.factories import (
        build_dataset_factory,
        build_environment_factory,
    )
    from algotrading.api.workers.training_worker import DefaultTrainingExecutor

    root = project_root or Path(__file__).resolve().parents[4]
    data_dir = root / "data" / "api"
    jobs_path = data_dir / "training_jobs.json"
    models_dir = root / "data" / "models"

    if worker is None:
        resolved_api_config = api_config or APIConfig()
        data_service = create_default_data_acquisition_service(
            api_config=resolved_api_config,
            project_root=root,
        )
        data_service._supporting_registry = model_service.supporting_registry
        environment_factory = build_environment_factory(
            data_service=data_service,
            model_service=model_service,
            project_root=root,
        )
        dataset_factory = build_dataset_factory(
            data_service=data_service,
            model_service=model_service,
            project_root=root,
        )
        executor = DefaultTrainingExecutor(
            models_dir=models_dir,
            environment_factory=environment_factory,
            dataset_factory=dataset_factory,
        )
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

    from algotrading.api.services.app_state import get_or_create_state

    def _build() -> TrainingService:
        from algotrading.api.services.model_service import get_model_service

        model_service = get_model_service(request)
        ws_manager = request.app.state.ws_manager
        service = create_default_training_service(
            ws_manager=ws_manager,
            model_service=model_service,
            api_config=request.app.state.api_config,
        )

        # Start the worker if there is a running event loop available.
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None
        if loop is not None and loop.is_running():
            asyncio.ensure_future(service.start(), loop=loop)
        return service

    return get_or_create_state(request, "training_service", _build)


def _artifact_exists(model_path: str) -> bool:
    """Return whether a saved RL artifact exists at ``model_path``.

    SB3 persists models with a ``.zip`` extension; some recorded paths omit it,
    so accept either form.
    """

    path = Path(model_path)
    return path.exists() or path.with_suffix(".zip").exists()


__all__ = [
    "TrainingService",
    "TrainingServiceError",
    "TrainingJobNotFoundError",
    "TrainingJobStateError",
    "create_default_training_service",
    "get_training_service",
]
