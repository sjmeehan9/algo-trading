"""Background worker that executes queued model training jobs."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Any, Callable, Protocol, runtime_checkable

from algotrading.api.schemas.models import ModelConfigResponse, ModelType
from algotrading.api.schemas.training import (
    TrainingJob,
    TrainingJobCreate,
)
from algotrading.src.models.tracking import (
    EvaluationMetrics,
    GenerationTracker,
    TrainingMetrics,
)

logger = logging.getLogger(__name__)


class TrainingExecutorError(Exception):
    """Base exception raised by training executor implementations."""


class TrainingExecutorConfigurationError(TrainingExecutorError):
    """Raised when the executor is not configured to run a given job."""


class TrainingCancelledError(TrainingExecutorError):
    """Raised by an executor when a job is cooperatively cancelled."""


@dataclass(slots=True)
class TrainingProgressUpdate:
    """Progress payload emitted by executors during training."""

    current_timestep: int
    total_timesteps: int
    current_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainingExecutionResult:
    """Final result returned by an executor on successful training."""

    timesteps_trained: int
    final_metrics: dict[str, Any] = field(default_factory=dict)
    training_metrics: TrainingMetrics | None = None
    evaluation_metrics: EvaluationMetrics | None = None
    model_path: str | None = None


@dataclass(slots=True)
class TrainingJobContext:
    """Mutable context object passed to a `TrainingExecutor`."""

    job: TrainingJob
    request: TrainingJobCreate
    model: ModelConfigResponse
    cancellation_event: Event
    progress_callback: Callable[[TrainingProgressUpdate], None]
    generation_tracker: GenerationTracker


@runtime_checkable
class TrainingExecutor(Protocol):
    """Protocol for components that perform actual training execution."""

    def execute(self, context: TrainingJobContext) -> TrainingExecutionResult:
        """Run the configured training job and return its result."""


# Environment factories accept ``(model, data_config)`` and may optionally
# accept a keyword ``cancellation`` event used to interrupt long data sourcing.
EnvironmentFactory = Callable[..., Any]
DatasetFactory = Callable[[ModelConfigResponse, dict[str, Any]], tuple[Any, Any]]


def _factory_accepts_cancellation(factory: Callable[..., Any]) -> bool:
    """Return whether an environment factory accepts a ``cancellation`` kwarg."""

    try:
        parameters = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return False
    if "cancellation" in parameters:
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


class DefaultTrainingExecutor:
    """Default executor wiring API training requests to existing trainers.

    The executor delegates environment and dataset construction to caller
    supplied factories so the API layer can stay decoupled from data and
    environment construction concerns. When a factory is missing for a model
    type a `TrainingExecutorConfigurationError` is raised so the failure is
    visible via the standard job failure path.
    """

    DEFAULT_TIMESTEPS = 100_000

    def __init__(
        self,
        models_dir: str | Path,
        environment_factory: EnvironmentFactory | None = None,
        dataset_factory: DatasetFactory | None = None,
    ) -> None:
        """Initialize the executor with persistence and factory dependencies.

        Args:
            models_dir: Directory to which trained model artifacts are saved.
            environment_factory: Callable returning a gymnasium environment for
                an RL training job.
            dataset_factory: Callable returning ``(X, y)`` arrays for an ML
                training job.
        """

        self._models_dir = Path(models_dir)
        self._environment_factory = environment_factory
        self._dataset_factory = dataset_factory

    def execute(self, context: TrainingJobContext) -> TrainingExecutionResult:
        """Execute training for the configured model."""

        model_type = context.model.model_type
        if model_type in (ModelType.CORE_RL, ModelType.SUPPORTING_RL):
            return self._execute_rl(context)
        if model_type == ModelType.SUPPORTING_ML:
            return self._execute_ml(context)
        raise TrainingExecutorConfigurationError(
            f"Unsupported model type for training: {model_type.value}"
        )

    def _build_environment(
        self,
        model: ModelConfigResponse,
        data_config: dict[str, Any],
        cancellation: Event,
    ) -> Any:
        """Invoke the environment factory, passing cancellation when supported.

        The default factory accepts a ``cancellation`` keyword so long-running
        data sourcing can be interrupted; simpler factories (used in tests) take
        only ``(model, data_config)`` and are called without it.
        """

        factory = self._environment_factory
        if factory is None:
            raise TrainingExecutorConfigurationError(
                "environment_factory must be configured to train RL models via API"
            )
        if _factory_accepts_cancellation(factory):
            return factory(model, data_config, cancellation=cancellation)
        return factory(model, data_config)

    def _resolve_continue_artifact(
        self, context: TrainingJobContext
    ) -> str | None:
        """Return the artifact path to warm-start from, or ``None`` for fresh.

        The request's ``continue_from_generation_id`` is validated when the job
        is created; this re-resolves the saved artifact at execution time and
        fails loudly if it has since become unavailable.
        """

        generation_id = context.request.continue_from_generation_id
        if not generation_id:
            return None

        tracker = context.generation_tracker
        generation = tracker.get_generation(generation_id) if tracker else None
        if generation is None or not generation.model_path:
            raise TrainingExecutorConfigurationError(
                f"Cannot continue training: generation '{generation_id}' has no "
                "saved artifact"
            )
        return generation.model_path

    def _execute_rl(self, context: TrainingJobContext) -> TrainingExecutionResult:
        """Execute an RL training job using `StableBaselines3Trainer`."""

        if self._environment_factory is None:
            raise TrainingExecutorConfigurationError(
                "environment_factory must be configured to train RL models via API"
            )

        from algotrading.src.trainers.rl_trainer import TrainingConfig
        from algotrading.src.trainers.sb3_trainer import (
            SB3Algorithm,
            StableBaselines3Trainer,
        )

        algorithm_value = context.model.algorithm.lower()
        try:
            algorithm = SB3Algorithm(algorithm_value)
        except ValueError as exc:
            raise TrainingExecutorConfigurationError(
                f"Unsupported SB3 algorithm '{context.model.algorithm}'"
            ) from exc

        total_timesteps = int(context.job.total_timesteps or self.DEFAULT_TIMESTEPS)
        if total_timesteps < 1:
            raise TrainingExecutorConfigurationError(
                "total_timesteps must be a positive integer"
            )

        factory_data_config = dict(context.request.data_config)
        factory_data_config.setdefault("total_timesteps", total_timesteps)
        env = self._build_environment(
            context.model, factory_data_config, context.cancellation_event
        )

        hyperparameters = dict(context.model.hyperparameters)
        learning_rate = _coerce_optional_float(hyperparameters.get("learning_rate"))
        batch_size = _coerce_optional_int(hyperparameters.get("batch_size"))
        n_steps = _coerce_optional_int(hyperparameters.get("n_steps"))
        policy = _resolve_policy(context=context, env=env)
        custom_params = _training_custom_params(
            hyperparameters=hyperparameters,
            training_config=context.request.training_config,
        )

        config = TrainingConfig(
            total_timesteps=total_timesteps,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_steps=n_steps,
            tensorboard_log=context.request.training_config.get("tensorboard_log"),
            custom_params=custom_params,
        )

        trainer = StableBaselines3Trainer(algorithm=algorithm, policy=policy)
        continue_artifact = self._resolve_continue_artifact(context)
        if continue_artifact is not None:
            logger.info(
                "Continuing RL training | model_id=%s generation=%s artifact=%s",
                context.model.model_id,
                context.request.continue_from_generation_id,
                continue_artifact,
            )
            trainer.load(continue_artifact, env=env)
        else:
            trainer.create_model(env=env, config=config)

        cancellation = context.cancellation_event
        progress_callback = context.progress_callback

        def _trainer_callback(payload: dict[str, object]) -> bool:
            if cancellation.is_set():
                return False
            timesteps = int(payload.get("timesteps", 0) or 0)
            metrics = {
                "reward": payload.get("reward"),
                "episodes": payload.get("episodes"),
            }
            progress_callback(
                TrainingProgressUpdate(
                    current_timestep=timesteps,
                    total_timesteps=total_timesteps,
                    current_metrics=metrics,
                )
            )
            return True

        result = trainer.train(
            config=config,
            callback=_trainer_callback,
            reset_num_timesteps=continue_artifact is None,
        )

        if cancellation.is_set():
            raise TrainingCancelledError("Training cancelled by request")

        model_path = self._save_model_artifact(
            trainer=trainer, context=context, suffix="zip"
        )

        training_metrics = TrainingMetrics(
            final_reward=float(result.final_reward),
            mean_reward=float(result.final_reward),
            std_reward=0.0,
            episodes_completed=int(result.episodes_completed),
            timesteps_trained=int(result.timesteps_trained),
            training_time_seconds=float(result.training_time_seconds),
        )

        return TrainingExecutionResult(
            timesteps_trained=int(result.timesteps_trained),
            final_metrics={
                "final_reward": training_metrics.final_reward,
                "episodes_completed": training_metrics.episodes_completed,
                "timesteps_trained": training_metrics.timesteps_trained,
                "training_time_seconds": training_metrics.training_time_seconds,
            },
            training_metrics=training_metrics,
            model_path=str(model_path) if model_path else None,
        )

    def _execute_ml(self, context: TrainingJobContext) -> TrainingExecutionResult:
        """Execute an ML training job using `NewsSentimentTrainer`."""

        if self._dataset_factory is None:
            raise TrainingExecutorConfigurationError(
                "dataset_factory must be configured to train ML models via API"
            )

        from algotrading.src.models.supporting.sentiment.news_sentiment import (
            NewsSentimentTrainer,
        )
        from algotrading.src.trainers.ml_trainer import MLTrainingConfig

        features, targets = self._dataset_factory(
            context.model, context.request.data_config
        )

        hyperparameters = dict(context.model.hyperparameters)
        epochs = _coerce_optional_int(hyperparameters.get("epochs")) or 1
        batch_size = _coerce_optional_int(hyperparameters.get("batch_size"))
        validation_split = _coerce_optional_float(
            hyperparameters.get("validation_split")
        )

        ml_config = MLTrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            early_stopping=bool(hyperparameters.get("early_stopping", False)),
            custom_params=context.request.training_config.get("custom_params"),
        )

        trainer = NewsSentimentTrainer(model_id=context.model.model_id)

        if context.cancellation_event.is_set():
            raise TrainingCancelledError("Training cancelled by request")

        # Emit a single in-progress update so callers can observe activity.
        context.progress_callback(
            TrainingProgressUpdate(
                current_timestep=0,
                total_timesteps=epochs,
                current_metrics={"phase": "fit_calibration"},
            )
        )

        result = trainer.train(features, targets, ml_config)

        if context.cancellation_event.is_set():
            raise TrainingCancelledError("Training cancelled by request")

        model_path = self._save_supporting_ml_artifact(
            trainer=trainer,
            context=context,
        )

        context.progress_callback(
            TrainingProgressUpdate(
                current_timestep=epochs,
                total_timesteps=epochs,
                current_metrics={"final_loss": float(result.final_loss)},
            )
        )

        training_metrics = TrainingMetrics(
            final_reward=float(result.final_loss),
            mean_reward=float(result.final_loss),
            std_reward=0.0,
            episodes_completed=int(result.epochs_trained),
            timesteps_trained=int(result.epochs_trained),
            training_time_seconds=float(result.training_time_seconds),
            custom_metrics=(
                {key: float(value) for key, value in (result.metrics or {}).items()}
                if result.metrics is not None
                else None
            ),
        )

        return TrainingExecutionResult(
            timesteps_trained=int(result.epochs_trained),
            final_metrics={
                "final_loss": float(result.final_loss),
                "epochs_trained": int(result.epochs_trained),
                "training_time_seconds": float(result.training_time_seconds),
            },
            training_metrics=training_metrics,
            model_path=str(model_path),
        )

    def _save_supporting_ml_artifact(
        self,
        *,
        trainer: Any,
        context: TrainingJobContext,
    ) -> Path:
        """Persist a supporting ML trainer artifact directory."""

        target_dir = self._models_dir / context.model.model_id / context.job.job_id
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            trainer.save(str(target_dir))
        except Exception as exc:
            raise TrainingExecutorError(
                "Failed saving trained supporting ML artifact for "
                f"job_id={context.job.job_id}"
            ) from exc
        return target_dir

    def _save_model_artifact(
        self,
        *,
        trainer: Any,
        context: TrainingJobContext,
        suffix: str,
    ) -> Path | None:
        """Persist a trained model artifact to the configured directory."""

        try:
            target_dir = self._models_dir / context.model.model_id
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / f"{context.job.job_id}.{suffix}"
            trainer.save(str(target_path.with_suffix("")))
        except Exception:
            logger.exception(
                "Failed saving trained model artifact for job_id=%s",
                context.job.job_id,
            )
            return None
        return target_path


class TrainingWorker:
    """Asyncio-based background worker that processes one job at a time."""

    def __init__(
        self,
        executor: TrainingExecutor,
        run_in_executor: Callable[..., Any] | None = None,
    ) -> None:
        """Initialize the worker with an executor implementation.

        Args:
            executor: The `TrainingExecutor` used to run individual jobs.
            run_in_executor: Optional callable to override the default
                ``asyncio.to_thread`` used to run blocking executor calls. Used
                primarily by tests to keep execution synchronous.
        """

        self._executor = executor
        self._run_in_executor = run_in_executor or asyncio.to_thread
        self._task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event = asyncio.Event()
        self._service: Any | None = None

    @property
    def executor(self) -> TrainingExecutor:
        """Return the configured training executor."""

        return self._executor

    @property
    def is_running(self) -> bool:
        """Return whether the background worker loop is active."""

        return self._task is not None and not self._task.done()

    def attach(self, service: Any) -> None:
        """Attach the worker to its owning training service."""

        self._service = service

    def start(self) -> None:
        """Start the background worker loop on the running event loop."""

        if self._task is not None and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="training-worker")

    async def stop(self) -> None:
        """Signal the worker to stop and wait for it to finish."""

        self._stop_event.set()
        if self._service is not None:
            self._service.notify_shutdown()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                self._task.cancel()
                try:
                    await self._task
                except (asyncio.CancelledError, Exception):
                    logger.warning("Training worker did not stop cleanly")
            self._task = None

    async def _run(self) -> None:
        """Main worker loop: wait for queued jobs and execute them."""

        if self._service is None:
            raise RuntimeError("TrainingWorker.attach() must be called before start()")

        service = self._service
        while not self._stop_event.is_set():
            job_id = await service.wait_for_next_job(self._stop_event)
            if job_id is None:
                break

            await self._process_job(service, job_id)

    async def _process_job(self, service: Any, job_id: str) -> None:
        """Run a single job to completion or terminal failure state."""

        cancellation_event: Event = service.get_cancellation_event(job_id)
        prepared = await service.prepare_job_execution(job_id)
        if prepared is None:
            return

        job, request, model = prepared

        loop = asyncio.get_running_loop()

        progress_updates: list[TrainingProgressUpdate] = []

        def _progress_callback(update: TrainingProgressUpdate) -> None:
            progress_updates.append(update)
            asyncio.run_coroutine_threadsafe(
                service.handle_progress(job_id, update), loop
            )

        context = TrainingJobContext(
            job=job,
            request=request,
            model=model,
            cancellation_event=cancellation_event,
            progress_callback=_progress_callback,
            generation_tracker=service.generation_tracker,
        )

        start_perf = time.perf_counter()
        try:
            result = await self._run_in_executor(self._executor.execute, context)
        except TrainingCancelledError:
            await service.mark_cancelled(job_id)
            return
        except Exception as exc:
            logger.exception("Training job %s failed", job_id)
            await service.mark_failed(job_id, str(exc))
            return

        elapsed = time.perf_counter() - start_perf
        await service.mark_completed(job_id, result=result, elapsed_seconds=elapsed)


def _coerce_optional_int(value: object) -> int | None:
    """Coerce a hyperparameter value to int when possible."""

    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _coerce_optional_float(value: object) -> float | None:
    """Coerce a hyperparameter value to float when possible."""

    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _resolve_policy(*, context: TrainingJobContext, env: Any) -> str:
    """Resolve the SB3 policy name for the model and environment shape."""

    raw_policy = (
        context.request.training_config.get("policy")
        or context.request.training_config.get("model_policy")
        or context.model.hyperparameters.get("policy")
        or context.model.hyperparameters.get("model_policy")
    )
    if raw_policy is not None:
        return str(raw_policy)

    try:
        from gymnasium.spaces import Dict as DictSpace
    except Exception:
        return "MlpPolicy"

    if isinstance(getattr(env, "observation_space", None), DictSpace):
        return "MultiInputPolicy"
    return "MlpPolicy"


def _training_custom_params(
    *,
    hyperparameters: dict[str, Any],
    training_config: dict[str, Any],
) -> dict[str, object] | None:
    """Merge model and job SB3 custom parameters."""

    reserved = {
        "learning_rate",
        "batch_size",
        "n_steps",
        "policy",
        "model_policy",
        "total_timesteps",
    }
    params = {
        str(key): value
        for key, value in hyperparameters.items()
        if key not in reserved and value is not None
    }
    raw_custom = training_config.get("custom_params")
    if isinstance(raw_custom, dict):
        params.update({str(key): value for key, value in raw_custom.items()})
    return params or None
