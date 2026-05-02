"""Deployment validation gates for production trading."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Protocol

from algotrading.src.trading.deployment.audit import DeploymentAuditLog
from algotrading.src.trainers import RLTrainer

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Model categories relevant to live-trading deployment eligibility."""

    CORE_RL = "core_rl"
    SUPPORTING_ML = "supporting_ml"
    SUPPORTING_RL = "supporting_rl"
    STRATEGY = "strategy"


class InvalidDeploymentError(Exception):
    """Raised when a model or runtime object is not eligible for deployment."""


class _ModelService(Protocol):
    """Minimal model-service protocol required by the deployment validator."""

    def get_model(self, model_id: str) -> object:
        """Return a model record by identifier."""

    def list_models(self, page: int = 1, page_size: int = 100) -> object:
        """Return a page of model records."""


class DeploymentValidator:
    """Validate that only trained core RL models can be deployed for trading."""

    DEPLOYABLE_TYPES = frozenset({ModelType.CORE_RL})
    DEPLOYABLE_GENERATION_STATUSES = frozenset({"completed", "evaluated"})

    def __init__(
        self,
        model_service: _ModelService,
        audit_log: DeploymentAuditLog,
    ) -> None:
        """Initialize the validator.

        Args:
            model_service: Service used to load model and generation metadata.
            audit_log: Append-only audit log for deployment attempts.
        """

        self.model_service = model_service
        self.audit_log = audit_log

    def validate_deployment(
        self,
        model_id: str,
        user_id: str | None = None,
        deployment_mode: str = "paper",
        generation_id: str | None = None,
        record_audit: bool = True,
    ) -> bool:
        """Validate that a model generation can be deployed for trading.

        Args:
            model_id: Model identifier requested for deployment.
            user_id: Optional user or client identifier for auditing.
            deployment_mode: Deployment mode such as ``paper`` or ``live``.
            generation_id: Optional generation identifier to validate.
            record_audit: Whether this call should write an audit entry.

        Returns:
            ``True`` when the deployment request is valid.

        Raises:
            InvalidDeploymentError: If the model or generation cannot be deployed.
        """

        mode = _normalize_deployment_mode(deployment_mode)
        model_type: str | None = None

        try:
            model = self.model_service.get_model(model_id)
        except Exception as exc:
            reason = f"Model not found: {model_id}"
            self._audit(
                record_audit=record_audit,
                model_id=model_id,
                user_id=user_id,
                mode=mode,
                result="rejected",
                reason=reason,
                model_type=None,
                generation_id=generation_id,
            )
            raise InvalidDeploymentError(reason) from exc

        try:
            normalized_type = normalize_model_type(getattr(model, "model_type"))
            model_type = normalized_type.value
        except InvalidDeploymentError as exc:
            reason = str(exc)
            self._audit(
                record_audit=record_audit,
                model_id=model_id,
                user_id=user_id,
                mode=mode,
                result="rejected",
                reason=reason,
                model_type=model_type,
                generation_id=generation_id,
            )
            raise

        if normalized_type not in self.DEPLOYABLE_TYPES:
            model_name = str(getattr(model, "name", model_id))
            reason = (
                f"Model type '{normalized_type.value}' cannot be deployed for trading. "
                "Only reinforcement learning models registered as core_rl can be "
                f"deployed. Model '{model_name}' is a {normalized_type.value} model."
            )
            self._audit(
                record_audit=record_audit,
                model_id=model_id,
                user_id=user_id,
                mode=mode,
                result="rejected",
                reason=reason,
                model_type=model_type,
                generation_id=generation_id,
            )
            raise InvalidDeploymentError(reason)

        generation = self._select_generation(model_id, generation_id)
        if generation is None:
            reason = "Model has not been trained. Deploy a trained model generation."
            self._audit(
                record_audit=record_audit,
                model_id=model_id,
                user_id=user_id,
                mode=mode,
                result="rejected",
                reason=reason,
                model_type=model_type,
                generation_id=generation_id,
            )
            raise InvalidDeploymentError(reason)

        generation_error = self._generation_error(generation)
        if generation_error is not None:
            self._audit(
                record_audit=record_audit,
                model_id=model_id,
                user_id=user_id,
                mode=mode,
                result="rejected",
                reason=generation_error,
                model_type=model_type,
                generation_id=_generation_id(generation, generation_id),
            )
            raise InvalidDeploymentError(generation_error)

        self._audit(
            record_audit=record_audit,
            model_id=model_id,
            user_id=user_id,
            mode=mode,
            result="approved",
            reason=None,
            model_type=model_type,
            generation_id=_generation_id(generation, generation_id),
        )
        return True

    def get_deployable_models(self) -> list[dict[str, object]]:
        """Return trained core RL models eligible for deployment."""

        deployable: list[dict[str, object]] = []
        for model in self._list_models():
            try:
                if (
                    normalize_model_type(getattr(model, "model_type"))
                    != ModelType.CORE_RL
                ):
                    continue
            except InvalidDeploymentError:
                logger.debug("Skipping model with unknown deployment type: %r", model)
                continue

            model_id = str(getattr(model, "model_id"))
            generation = self._select_generation(model_id, generation_id=None)
            if generation is None or self._generation_error(generation) is not None:
                continue

            deployable.append(
                {
                    "model_id": model_id,
                    "name": str(getattr(model, "name", model_id)),
                    "latest_generation": _generation_id(generation, None),
                    "metrics": _generation_metrics(generation),
                }
            )
        return deployable

    def _select_generation(
        self,
        model_id: str,
        generation_id: str | None,
    ) -> object | None:
        """Select an explicit generation or the newest deployable generation."""

        if generation_id is not None:
            generation = self._get_generation_by_id(generation_id)
            if (
                generation is None
                or str(getattr(generation, "model_id", "")) != model_id
            ):
                return None
            return generation

        generations = self._get_generations_for_model(model_id)
        generations.sort(key=_generation_sort_key, reverse=True)
        for generation in generations:
            if self._generation_error(generation) is None:
                return generation
        return generations[0] if generations else None

    def _get_generation_by_id(self, generation_id: str) -> object | None:
        """Return a generation record from the backing model service."""

        generation_tracker = getattr(self.model_service, "generation_tracker", None)
        if generation_tracker is not None:
            return generation_tracker.get_generation(generation_id)

        getter = getattr(self.model_service, "get_generation_detail_by_id", None)
        if callable(getter):
            try:
                return getter(generation_id)
            except Exception:
                return None
        return None

    def _get_generations_for_model(self, model_id: str) -> list[object]:
        """Return generation records for a model from available service APIs."""

        generation_tracker = getattr(self.model_service, "generation_tracker", None)
        if generation_tracker is not None:
            return list(generation_tracker.get_generations_for_model(model_id))

        getter = getattr(self.model_service, "get_generations", None)
        if not callable(getter):
            return []

        try:
            response = getter(model_id=model_id, page=1, page_size=100)
        except TypeError:
            response = getter(model_id)
        return list(getattr(response, "items", response or []))

    def _list_models(self) -> list[object]:
        """List all models across paginated service responses."""

        models: list[object] = []
        page = 1
        page_size = 100
        while True:
            response = self.model_service.list_models(page=page, page_size=page_size)
            items = list(getattr(response, "items", response or []))
            models.extend(items)
            pages = int(getattr(response, "pages", 1) or 1)
            if page >= pages:
                break
            page += 1
        return models

    def _generation_error(self, generation: object) -> str | None:
        """Return the blocking reason for a generation, if one exists."""

        status = str(getattr(generation, "status", "")).strip().lower()
        if status and status not in self.DEPLOYABLE_GENERATION_STATUSES:
            return (
                "Selected generation must be completed or evaluated before deployment."
            )

        if hasattr(generation, "model_path") and not getattr(generation, "model_path"):
            return "Selected generation has no saved model artifact path."

        return None

    def _audit(
        self,
        *,
        record_audit: bool,
        model_id: str,
        user_id: str | None,
        mode: str,
        result: str,
        reason: str | None,
        model_type: str | None,
        generation_id: str | None,
    ) -> None:
        """Record an audit event when requested."""

        if not record_audit:
            return
        self.audit_log.log_attempt(
            model_id=model_id,
            user_id=user_id,
            mode=mode,
            result=result,
            reason=reason,
            model_type=model_type,
            generation_id=generation_id,
        )


def normalize_model_type(value: object) -> ModelType:
    """Normalize a raw model type enum or string into ``ModelType``."""

    raw_value = getattr(value, "value", value)
    normalized = str(raw_value).strip().lower()
    try:
        return ModelType(normalized)
    except ValueError as exc:
        raise InvalidDeploymentError(
            f"Unknown model type '{raw_value}' for deployment validation."
        ) from exc


def validate_core_model_instance(
    core_model: object,
    configured_model_type: object | None = None,
) -> None:
    """Validate that a runtime core model is an RL trainer instance.

    Args:
        core_model: Runtime model object passed to live inference.
        configured_model_type: Optional deployment model type from configuration.

    Raises:
        InvalidDeploymentError: If the object or configuration is not core RL.
    """

    if configured_model_type is not None:
        normalized_type = normalize_model_type(configured_model_type)
        if normalized_type != ModelType.CORE_RL:
            raise InvalidDeploymentError(
                f"Configured model type '{normalized_type.value}' cannot be used as "
                "the live-trading core model. Only core_rl is deployable."
            )

    if not isinstance(core_model, RLTrainer):
        raise InvalidDeploymentError(
            "The live-trading core model must implement RLTrainer. Supporting ML "
            "models and strategies cannot execute trades directly."
        )


def validate_pipeline_deployment_config(
    pipeline: dict[str, object],
    deployment_mode: str = "paper",
) -> None:
    """Validate legacy pipeline configuration before live trading starts."""

    del deployment_mode
    pipeline_root = pipeline.get("pipeline", pipeline)
    if not isinstance(pipeline_root, dict):
        raise InvalidDeploymentError("Pipeline configuration must be a dictionary.")

    pipeline_type = str(pipeline_root.get("pipeline_type", "")).strip().lower()
    if pipeline_type and pipeline_type != "rl":
        raise InvalidDeploymentError(
            f"Pipeline type '{pipeline_type}' cannot be deployed for live trading. "
            "Only core RL pipelines can execute trades."
        )

    configured_type = _extract_configured_deployment_type(pipeline_root)
    if configured_type is not None:
        normalized_type = normalize_model_type(configured_type)
        if normalized_type != ModelType.CORE_RL:
            raise InvalidDeploymentError(
                f"Configured model type '{normalized_type.value}' cannot be deployed "
                "for live trading. Only core_rl is deployable."
            )


def known_model_type(value: object) -> bool:
    """Return whether a raw value names a deployment model category."""

    try:
        normalize_model_type(value)
    except InvalidDeploymentError:
        return False
    return True


def _normalize_deployment_mode(value: str) -> str:
    """Normalize a deployment mode string."""

    mode = value.strip().lower()
    if mode not in {"paper", "live"}:
        raise InvalidDeploymentError("deployment_mode must be 'paper' or 'live'.")
    return mode


def _extract_configured_deployment_type(
    pipeline_root: dict[str, object],
) -> object | None:
    """Extract an explicit deployment model category from pipeline config."""

    for key in ("deployment_model_type", "model_category", "root_model_type"):
        value = pipeline_root.get(key)
        if value is not None:
            return value

    root_model_type = pipeline_root.get("model_type")
    if root_model_type is not None and known_model_type(root_model_type):
        return root_model_type

    model_section = pipeline_root.get("model")
    if isinstance(model_section, dict):
        for key in ("deployment_model_type", "model_category", "root_model_type"):
            value = model_section.get(key)
            if value is not None:
                return value
        model_type = model_section.get("model_type")
        if model_type is not None and known_model_type(model_type):
            return model_type

    return None


def _generation_id(generation: object, fallback: str | None) -> str | None:
    """Return a generation identifier from a record or fallback value."""

    value = getattr(generation, "generation_id", fallback)
    return str(value) if value is not None else None


def _generation_metrics(generation: object) -> dict[str, object]:
    """Return JSON-safe metrics from a generation-like object."""

    metrics = getattr(generation, "metrics", None)
    if isinstance(metrics, dict):
        return dict(metrics)

    payload: dict[str, object] = {}
    for attribute in ("training_metrics", "evaluation_metrics"):
        metric_obj = getattr(generation, attribute, None)
        if metric_obj is None:
            continue
        to_dict = getattr(metric_obj, "to_dict", None)
        metric_payload = to_dict() if callable(to_dict) else vars(metric_obj)
        if isinstance(metric_payload, dict):
            payload.update(
                {
                    str(key): value
                    for key, value in metric_payload.items()
                    if isinstance(value, (int, float, str, bool))
                }
            )
    return payload


def _generation_sort_key(generation: object) -> tuple[int, str]:
    """Return a stable sort key for newest-generation selection."""

    generation_number = int(getattr(generation, "generation_number", 0) or 0)
    created_at = getattr(generation, "created_at", "")
    return generation_number, str(created_at)


__all__ = [
    "DeploymentValidator",
    "InvalidDeploymentError",
    "ModelType",
    "known_model_type",
    "normalize_model_type",
    "validate_core_model_instance",
    "validate_pipeline_deployment_config",
]
