"""Pre-deployment candidate discovery, validation, and persistence."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import RLock

from algotrading.api.schemas.backtesting import BacktestResult, BacktestStatus
from algotrading.api.schemas.deployment import (
    DeployableModel,
    DeploymentBacktestSummary,
    DeploymentCandidate,
    DeploymentGenerationSummary,
    DeploymentReadiness,
    DeploymentSelection,
    DeploymentValidationRequest,
    ReadinessCheck,
    ReadinessCheckStatus,
)
from algotrading.api.schemas.models import ModelConfigResponse, ModelType
from algotrading.api.services.backtest_service import BacktestService
from algotrading.api.services.model_service import (
    GenerationNotFoundError,
    ModelNotFoundError,
    ModelService,
)
from algotrading.src.models.registry import ModelState
from algotrading.src.models.tracking import EvaluationMetrics, Generation
from algotrading.src.trading.deployment import (
    DeploymentAuditLog,
    DeploymentValidator,
    InvalidDeploymentError,
)
from fastapi import Request

logger = logging.getLogger(__name__)


class DeploymentServiceError(Exception):
    """Base error class for deployment selection operations."""


class DeploymentValidationError(DeploymentServiceError):
    """Raised when a deployment selection is invalid or not deployable."""


class DeploymentSelectionNotFoundError(DeploymentServiceError):
    """Raised when no deployment selection has been persisted."""


class DeploymentService:
    """Service for Phase 5 pre-deployment model generation selection.

    The service intentionally validates and persists only candidate metadata. It
    does not open broker connections, start paper sessions, or execute live
    inference.
    """

    STALE_EVALUATION_DAYS = 30
    OPTIONAL_METRICS = ("sharpe_ratio", "max_drawdown", "total_return")

    def __init__(
        self,
        model_service: ModelService,
        backtest_service: BacktestService | None = None,
        selection_path: str | Path | None = None,
        audit_log: DeploymentAuditLog | None = None,
    ) -> None:
        """Initialize service dependencies.

        Args:
            model_service: Model/generation registry business service.
            backtest_service: Optional service used to discover stored backtests.
            selection_path: Optional JSON path for persisted selected candidate.
            audit_log: Optional deployment audit log override.
        """

        self.model_service = model_service
        self.backtest_service = backtest_service
        self._selection_path = Path(selection_path) if selection_path else None
        audit_path = (
            self._selection_path.parent / "deployment_audit.jsonl"
            if self._selection_path is not None
            else None
        )
        self.audit_log = audit_log or DeploymentAuditLog(log_path=audit_path)
        self.deployment_validator = DeploymentValidator(
            model_service=self.model_service,
            audit_log=self.audit_log,
        )
        self._lock = RLock()

    def list_candidates(self) -> list[DeploymentCandidate]:
        """Return all core RL models eligible for pre-deployment review."""

        candidates = [
            self._build_candidate(model) for model in self._list_core_models()
        ]
        candidates.sort(key=lambda candidate: candidate.name.lower())
        return candidates

    def _list_core_models(self) -> list[ModelConfigResponse]:
        """Return every core RL model across paginated model-service results."""

        page = 1
        page_size = 100
        models: list[ModelConfigResponse] = []
        while True:
            response = self.model_service.list_models(
                model_type=ModelType.CORE_RL,
                page=page,
                page_size=page_size,
            )
            models.extend(response.items)
            if response.pages == 0 or page >= response.pages:
                break
            page += 1
        return models

    def validate_selection(
        self, request: DeploymentValidationRequest
    ) -> DeploymentReadiness:
        """Validate a selected core RL model generation."""

        model, generation = self._resolve_selection(request)
        return self._evaluate_readiness(model=model, generation=generation)

    def list_deployable_models(self) -> list[DeployableModel]:
        """Return trained core RL models that pass hard deployment validation."""

        return [
            DeployableModel.model_validate(item)
            for item in self.deployment_validator.get_deployable_models()
        ]

    def save_selection(
        self,
        request: DeploymentValidationRequest,
        selected_by: str | None = None,
    ) -> DeploymentSelection:
        """Persist a validated deployment candidate selection."""

        logged = False
        try:
            model, generation = self._resolve_selection(request)
            self.deployment_validator.validate_deployment(
                model_id=request.model_id,
                user_id=selected_by,
                deployment_mode="paper",
                generation_id=request.generation_id,
                record_audit=False,
            )
            readiness = self._evaluate_readiness(model=model, generation=generation)
            if not readiness.deployable:
                details = "; ".join(readiness.errors) or "Candidate is not deployable"
                self._log_deployment_attempt(
                    request=request,
                    selected_by=selected_by,
                    result="rejected",
                    reason=details,
                    model_type=_model_type_value(model.model_type),
                )
                logged = True
                raise DeploymentValidationError(details)
        except InvalidDeploymentError as exc:
            if not logged:
                self._log_deployment_attempt(
                    request=request,
                    selected_by=selected_by,
                    result="rejected",
                    reason=str(exc),
                    model_type=self._safe_model_type(request.model_id),
                )
            raise DeploymentValidationError(str(exc)) from exc
        except (
            ModelNotFoundError,
            GenerationNotFoundError,
            DeploymentValidationError,
        ) as exc:
            if not logged:
                self._log_deployment_attempt(
                    request=request,
                    selected_by=selected_by,
                    result="rejected",
                    reason=str(exc),
                    model_type=self._safe_model_type(request.model_id),
                )
            raise

        candidate = self._build_candidate(
            model, selected_generation_id=generation.generation_id
        )
        selection = DeploymentSelection(
            model_id=model.model_id,
            generation_id=generation.generation_id,
            selected_at=datetime.now(tz=UTC),
            candidate=candidate,
            readiness=readiness,
            selected_by=selected_by,
        )
        self._persist_selection(selection)
        self._log_deployment_attempt(
            request=request,
            selected_by=selected_by,
            result="approved",
            reason=None,
            model_type=_model_type_value(model.model_type),
        )
        return selection

    def get_selection(self) -> DeploymentSelection:
        """Return the persisted deployment candidate selection."""

        if self._selection_path is None or not self._selection_path.exists():
            raise DeploymentSelectionNotFoundError(
                "No deployment selection has been saved"
            )

        try:
            payload = json.loads(self._selection_path.read_text(encoding="utf-8"))
            selection = DeploymentSelection.model_validate(payload)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise DeploymentServiceError(
                f"Deployment selection store is invalid: {exc}"
            ) from exc

        self.validate_selection(
            DeploymentValidationRequest(
                model_id=selection.model_id,
                generation_id=selection.generation_id,
            )
        )
        return selection

    def clear_selection(self) -> None:
        """Delete the persisted deployment selection if present."""

        if self._selection_path is None:
            return
        with self._lock:
            try:
                self._selection_path.unlink(missing_ok=True)
            except OSError as exc:
                raise DeploymentServiceError(
                    f"Could not clear deployment selection: {exc}"
                ) from exc

    def _resolve_selection(
        self, request: DeploymentValidationRequest
    ) -> tuple[ModelConfigResponse, Generation]:
        """Load and type-check the selected model and generation."""

        model = self.model_service.get_model(request.model_id)
        if model.model_type != ModelType.CORE_RL:
            raise DeploymentValidationError(
                "Only core RL models can be selected for live-trading deployment"
            )

        generation = self.model_service.generation_tracker.get_generation(
            request.generation_id
        )
        if generation is None or generation.model_id != model.model_id:
            raise GenerationNotFoundError(
                f"Generation '{request.generation_id}' not found for model '{model.model_id}'"
            )
        return model, generation

    def _build_candidate(
        self,
        model: ModelConfigResponse,
        selected_generation_id: str | None = None,
    ) -> DeploymentCandidate:
        """Build one candidate summary for the deployment UI."""

        generations = self.model_service.generation_tracker.get_generations_for_model(
            model.model_id
        )
        generations.sort(key=lambda item: item.generation_number, reverse=True)
        selected_generation = self._select_generation(
            generations, selected_generation_id
        )
        readiness = (
            self._evaluate_readiness(model=model, generation=selected_generation)
            if selected_generation is not None
            else self._readiness_without_generation(model)
        )
        summaries = [self._generation_summary(item) for item in generations]
        latest_backtest = summaries[0].evaluation if summaries else None

        return DeploymentCandidate(
            model_id=model.model_id,
            name=model.name,
            description=model.description,
            algorithm=model.algorithm,
            trainer_type=model.trainer_type,
            state=model.state,
            supporting_model_ids=list(model.supporting_model_ids),
            strategy_ids=list(model.strategy_ids),
            available_generations=summaries,
            latest_backtest=latest_backtest,
            readiness=readiness,
        )

    def _select_generation(
        self, generations: list[Generation], selected_generation_id: str | None
    ) -> Generation | None:
        """Pick the selected generation or newest available generation."""

        if selected_generation_id is not None:
            return next(
                (
                    generation
                    for generation in generations
                    if generation.generation_id == selected_generation_id
                ),
                None,
            )
        return generations[0] if generations else None

    def _readiness_without_generation(
        self, model: ModelConfigResponse
    ) -> DeploymentReadiness:
        """Return candidate readiness when no generation exists yet."""

        builder = _ReadinessBuilder()
        builder.add(
            key="model_type",
            label="Core RL model",
            ok=model.model_type == ModelType.CORE_RL,
            passed="Model is a core RL deployment root.",
            failed="Only core RL models can be selected for deployment.",
        )
        builder.fail(
            key="generation",
            label="Trained generation",
            message="Select a completed and evaluated generation before deployment.",
        )
        self._validate_supporting_inputs(model, builder)
        self._validate_strategy_inputs(model, builder)
        builder.warn(
            key="optimizer_history",
            label="Optimizer history",
            message="No optimizer history was found; manual hyperparameter review is recommended.",
        )
        return builder.build()

    def _evaluate_readiness(
        self,
        *,
        model: ModelConfigResponse,
        generation: Generation,
    ) -> DeploymentReadiness:
        """Evaluate deployment readiness for one selected generation."""

        builder = _ReadinessBuilder()
        builder.add(
            key="model_type",
            label="Core RL model",
            ok=model.model_type == ModelType.CORE_RL,
            passed="Model is a core RL deployment root.",
            failed="Only core RL models can be selected for deployment.",
        )
        builder.add(
            key="generation_exists",
            label="Generation record",
            ok=generation.model_id == model.model_id,
            passed="Selected generation belongs to this model.",
            failed="Selected generation does not belong to this model.",
        )
        builder.add(
            key="generation_completed",
            label="Training completed",
            ok=generation.status in {"completed", "evaluated"},
            passed=f"Generation status is {generation.status}.",
            failed="Generation must be completed or evaluated before deployment.",
        )
        builder.add(
            key="model_artifact",
            label="Model artifact",
            ok=bool(generation.model_path),
            passed="Generation has a saved model artifact path.",
            failed="Generation has no saved model artifact path.",
        )
        self._validate_evaluation_evidence(generation, builder)
        self._validate_supporting_inputs(model, builder)
        self._validate_strategy_inputs(model, builder)
        self._validate_optional_metrics(generation, builder)
        builder.warn(
            key="optimizer_history",
            label="Optimizer history",
            message="No optimizer history was found; manual hyperparameter review is recommended.",
        )
        return builder.build()

    def _validate_evaluation_evidence(
        self, generation: Generation, builder: _ReadinessBuilder
    ) -> None:
        """Require completed evaluation or backtest metrics."""

        evaluation = self._latest_evaluation(generation)
        builder.add(
            key="evaluation_evidence",
            label="Evaluation evidence",
            ok=evaluation is not None,
            passed="Completed evaluation/backtest metrics are available.",
            failed="A completed backtest or evaluation metric record is required.",
        )
        if evaluation is None:
            return
        if evaluation.completed_at is not None:
            cutoff = datetime.now(tz=UTC) - timedelta(days=self.STALE_EVALUATION_DAYS)
            if evaluation.completed_at < cutoff:
                builder.warn(
                    key="evaluation_freshness",
                    label="Evaluation freshness",
                    message="Latest evaluation is older than 30 days.",
                )

    def _validate_supporting_inputs(
        self, model: ModelConfigResponse, builder: _ReadinessBuilder
    ) -> None:
        """Validate referenced supporting models still exist and are ready."""

        if not model.supporting_model_ids:
            builder.add_pass(
                key="supporting_models",
                label="Supporting model inputs",
                message="No supporting model inputs are required.",
            )
            return

        for model_id in model.supporting_model_ids:
            entry = self.model_service.supporting_registry.get(model_id)
            if entry is None:
                builder.fail(
                    key=f"supporting_model:{model_id}",
                    label="Supporting model inputs",
                    message=f"Supporting model '{model_id}' is missing.",
                )
                continue
            builder.add(
                key=f"supporting_model:{model_id}",
                label="Supporting model inputs",
                ok=entry.state == ModelState.READY,
                passed=f"Supporting model '{model_id}' is ready.",
                failed=(
                    f"Supporting model '{model_id}' is not ready "
                    f"(current state: {entry.state.value})."
                ),
            )

    def _validate_strategy_inputs(
        self, model: ModelConfigResponse, builder: _ReadinessBuilder
    ) -> None:
        """Validate referenced strategy IDs still exist."""

        if not model.strategy_ids:
            builder.add_pass(
                key="strategies",
                label="Strategy inputs",
                message="No strategy inputs are required.",
            )
            return

        for strategy_id in model.strategy_ids:
            exists = self.model_service.strategy_registry.get(strategy_id) is not None
            builder.add(
                key=f"strategy:{strategy_id}",
                label="Strategy inputs",
                ok=exists,
                passed=f"Strategy '{strategy_id}' is registered.",
                failed=f"Strategy '{strategy_id}' is missing.",
            )

    def _validate_optional_metrics(
        self, generation: Generation, builder: _ReadinessBuilder
    ) -> None:
        """Warn when optional comparison metrics are absent."""

        missing = [
            metric
            for metric in self.OPTIONAL_METRICS
            if generation.metric_value(metric) is None
        ]
        if missing:
            builder.warn(
                key="optional_metrics",
                label="Comparison metrics",
                message=f"Optional comparison metrics are missing: {', '.join(missing)}.",
            )
        else:
            builder.add_pass(
                key="optional_metrics",
                label="Comparison metrics",
                message="Core comparison metrics are available.",
            )

    def _generation_summary(
        self, generation: Generation
    ) -> DeploymentGenerationSummary:
        """Convert a generation record to deployment-facing summary data."""

        training_duration = 0.0
        final_reward: float | None = None
        metrics: dict[str, float | int | str | bool | None] = {}

        if generation.training_metrics is not None:
            training_duration = generation.training_metrics.training_time_seconds
            final_reward = generation.training_metrics.final_reward
            metrics.update(
                {
                    "final_reward": generation.training_metrics.final_reward,
                    "mean_reward": generation.training_metrics.mean_reward,
                    "std_reward": generation.training_metrics.std_reward,
                    "episodes_completed": generation.training_metrics.episodes_completed,
                    "timesteps_trained": generation.training_metrics.timesteps_trained,
                }
            )
            if generation.training_metrics.custom_metrics:
                metrics.update(generation.training_metrics.custom_metrics)

        if generation.evaluation_metrics is not None:
            metrics.update(_evaluation_metrics_to_dict(generation.evaluation_metrics))

        return DeploymentGenerationSummary(
            generation_id=generation.generation_id,
            generation_number=generation.generation_number,
            status=generation.status,
            created_at=generation.created_at,
            training_duration_seconds=training_duration,
            final_reward=final_reward,
            model_path=generation.model_path,
            metrics=metrics,
            evaluation=self._latest_evaluation(generation),
        )

    def _latest_evaluation(
        self, generation: Generation
    ) -> DeploymentBacktestSummary | None:
        """Return latest backtest result or generation evaluation metrics."""

        result = self._latest_completed_backtest(generation)
        if result is not None and result.metrics is not None:
            return DeploymentBacktestSummary(
                source="backtest",
                backtest_id=result.backtest_id,
                completed_at=result.completed_at,
                metrics=result.metrics,
            )
        if generation.evaluation_metrics is not None:
            return DeploymentBacktestSummary(
                source="generation_evaluation",
                backtest_id=None,
                completed_at=None,
                metrics=_evaluation_metrics_to_dict(generation.evaluation_metrics),
            )
        return None

    def _latest_completed_backtest(
        self, generation: Generation
    ) -> BacktestResult | None:
        """Return newest completed backtest for one generation when available."""

        if self.backtest_service is None:
            return None
        try:
            response = self.backtest_service.list_results(
                model_id=generation.model_id,
                generation_id=generation.generation_id,
                result_status=BacktestStatus.COMPLETED,
                page=1,
                page_size=1,
            )
        except Exception:
            logger.exception(
                "Failed to load backtest evidence for generation %s",
                generation.generation_id,
            )
            return None
        return response.items[0] if response.items else None

    def _persist_selection(self, selection: DeploymentSelection) -> None:
        """Write selection JSON to disk when persistence is configured."""

        if self._selection_path is None:
            return
        with self._lock:
            self._selection_path.parent.mkdir(parents=True, exist_ok=True)
            self._selection_path.write_text(
                json.dumps(selection.model_dump(mode="json"), indent=2, sort_keys=True),
                encoding="utf-8",
            )

    def _safe_model_type(self, model_id: str) -> str | None:
        """Best-effort model type lookup for rejected audit events."""

        try:
            return _model_type_value(self.model_service.get_model(model_id).model_type)
        except Exception:
            return None

    def _log_deployment_attempt(
        self,
        *,
        request: DeploymentValidationRequest,
        selected_by: str | None,
        result: str,
        reason: str | None,
        model_type: str | None,
    ) -> None:
        """Record a persisted deployment-selection attempt."""

        self.audit_log.log_attempt(
            model_id=request.model_id,
            user_id=selected_by,
            mode="paper",
            result=result,
            reason=reason,
            model_type=model_type,
            generation_id=request.generation_id,
            endpoint="deployment_selection",
        )


class _ReadinessBuilder:
    """Collect readiness checks while preserving errors and warnings."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.checks: list[ReadinessCheck] = []

    def add(
        self,
        *,
        key: str,
        label: str,
        ok: bool,
        passed: str,
        failed: str,
    ) -> None:
        """Add a passed or failed check."""

        if ok:
            self.add_pass(key=key, label=label, message=passed)
            return
        self.fail(key=key, label=label, message=failed)

    def add_pass(self, *, key: str, label: str, message: str) -> None:
        """Add a passed readiness check."""

        self.checks.append(
            ReadinessCheck(
                key=key,
                label=label,
                status=ReadinessCheckStatus.PASSED,
                message=message,
            )
        )

    def fail(self, *, key: str, label: str, message: str) -> None:
        """Add a blocking readiness failure."""

        self.errors.append(message)
        self.checks.append(
            ReadinessCheck(
                key=key,
                label=label,
                status=ReadinessCheckStatus.FAILED,
                message=message,
            )
        )

    def warn(self, *, key: str, label: str, message: str) -> None:
        """Add a non-blocking readiness warning."""

        self.warnings.append(message)
        self.checks.append(
            ReadinessCheck(
                key=key,
                label=label,
                status=ReadinessCheckStatus.WARNING,
                message=message,
            )
        )

    def build(self) -> DeploymentReadiness:
        """Return the final readiness object."""

        return DeploymentReadiness(
            deployable=not self.errors,
            errors=list(self.errors),
            warnings=list(self.warnings),
            checks=list(self.checks),
        )


def _evaluation_metrics_to_dict(
    metrics: EvaluationMetrics,
) -> dict[str, float | int | None]:
    """Convert generation evaluation metrics to API-compatible primitives."""

    payload = metrics.to_dict()
    custom_metrics = payload.pop("custom_metrics", None)
    result: dict[str, float | int | None] = {
        str(key): value
        for key, value in payload.items()
        if isinstance(value, (int, float)) or value is None
    }
    if isinstance(custom_metrics, dict):
        result.update(
            {
                str(key): value
                for key, value in custom_metrics.items()
                if isinstance(value, (int, float)) or value is None
            }
        )
    return result


def _model_type_value(model_type: object) -> str:
    """Return a string value for model-type enums or raw strings."""

    return str(getattr(model_type, "value", model_type))


def create_default_deployment_service(
    *,
    model_service: ModelService,
    backtest_service: BacktestService | None = None,
    project_root: Path | None = None,
) -> DeploymentService:
    """Build a DeploymentService with filesystem-backed persistence."""

    root = project_root or Path(__file__).resolve().parents[4]
    selection_path = root / "data" / "api" / "deployment_selection.json"
    return DeploymentService(
        model_service=model_service,
        backtest_service=backtest_service,
        selection_path=selection_path,
    )


def get_deployment_service(request: Request) -> DeploymentService:
    """FastAPI dependency resolver for the shared DeploymentService instance."""

    service = getattr(request.app.state, "deployment_service", None)
    if service is None:
        from algotrading.api.services.backtest_service import get_backtest_service
        from algotrading.api.services.model_service import get_model_service

        model_service = get_model_service(request)
        backtest_service = get_backtest_service(request)
        service = create_default_deployment_service(
            model_service=model_service,
            backtest_service=backtest_service,
        )
        request.app.state.deployment_service = service
    return service


__all__ = [
    "DeploymentService",
    "DeploymentServiceError",
    "DeploymentValidationError",
    "DeploymentSelectionNotFoundError",
    "create_default_deployment_service",
    "get_deployment_service",
]
