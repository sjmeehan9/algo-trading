"""Lifecycle operations for supporting ML/RL models exposed through the API.

This service provides the operator-facing surface to make supporting models
ready for inference without hand-editing the registry JSON. It supports three
readiness paths required by Phase 7:

1. Activating a pretrained supporting ML backend (currently the news sentiment
   implementation) from model hyperparameters.
2. Loading and validating an externally trained artifact.
3. Unloading a model back to a non-ready state.

All state transitions are delegated to :class:`SupportingModelRegistry` so they
persist to ``data/api/supporting_models_registry.json``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from algotrading.api.schemas.models import ModelType
from algotrading.api.schemas.supporting_lifecycle import (
    ActivatePretrainedRequest,
    LoadArtifactRequest,
    ReadinessCheck,
    SupportingModelLifecycleResponse,
)
from algotrading.api.services.model_service import (
    ModelNotFoundError,
    ModelService,
)
from algotrading.src.models.registry import ModelEntry, ModelState, RegistryError
from algotrading.src.models.supporting.sentiment.config import (
    SentimentConfig,
    SentimentModelType,
)
from algotrading.src.models.supporting.sentiment.news_sentiment import (
    NewsSentimentTrainer,
)
from algotrading.src.trainers.sb3_trainer import SB3Algorithm, StableBaselines3Trainer

logger = logging.getLogger(__name__)

_SB3_TRAINER_CLASS = "algotrading.src.trainers.sb3_trainer.StableBaselines3Trainer"

# Algorithms that map onto a concrete pretrained sentiment backend. Anything
# outside this set (for example sklearn random_forest or tensorflow lstm) is not
# a real pretrained sentiment implementation and must be rejected rather than
# silently activated.
_SENTIMENT_ALGORITHM_BACKENDS: dict[str, SentimentModelType] = {
    "transformer_sentiment": SentimentModelType.FINBERT,
    "finbert": SentimentModelType.FINBERT,
    "vader": SentimentModelType.VADER,
    "provider": SentimentModelType.PROVIDER_PASSTHROUGH,
    "provider_passthrough": SentimentModelType.PROVIDER_PASSTHROUGH,
}

_SMOKE_TEXT = "Markets rallied today on strong earnings."


class SupportingLifecycleError(Exception):
    """Base error for supporting model lifecycle operations."""


class UnsupportedLifecycleModelError(SupportingLifecycleError):
    """Raised when an operation is invalid for the model's type/category."""


class LifecycleValidationError(SupportingLifecycleError):
    """Raised when lifecycle inputs are invalid (bad path, unsupported algo)."""


class LifecycleOperationError(SupportingLifecycleError):
    """Raised when a load/activate/smoke step fails at runtime."""


class SupportingModelLifecycleService:
    """Coordinate supporting model lifecycle transitions over the registry."""

    def __init__(self, model_service: ModelService) -> None:
        """Initialize the lifecycle service.

        Args:
            model_service: Shared model service exposing the supporting registry.
        """

        self._model_service = model_service
        self._registry = model_service.supporting_registry

    def get_lifecycle(self, model_id: str) -> SupportingModelLifecycleResponse:
        """Return a lifecycle snapshot for a supporting model.

        Args:
            model_id: Identifier of the supporting model.

        Returns:
            A lifecycle response describing state and readiness.

        Raises:
            ModelNotFoundError: If no model exists with the given ID.
            UnsupportedLifecycleModelError: If the model is a core RL model.
        """

        entry = self._require_supporting_entry(model_id)
        return self._build_response(entry)

    def activate_pretrained(
        self, model_id: str, request: ActivatePretrainedRequest
    ) -> SupportingModelLifecycleResponse:
        """Activate a pretrained supporting ML backend and mark it READY.

        Args:
            model_id: Identifier of the supporting model.
            request: Optional hyperparameter overrides for backend construction.

        Returns:
            The updated lifecycle response (READY on success).

        Raises:
            ModelNotFoundError: If the model does not exist.
            UnsupportedLifecycleModelError: If the model is not a supporting ML
                model or maps to an unsupported pretrained backend.
            LifecycleValidationError: If hyperparameters are invalid.
            LifecycleOperationError: If backend construction or the smoke
                prediction fails.
        """

        entry = self._require_supporting_entry(model_id)
        if entry.config.model_type != "ml":
            raise UnsupportedLifecycleModelError(
                "Pretrained activation is only supported for supporting_ml models; "
                f"model '{model_id}' is supporting_rl"
            )

        config_payload = dict(entry.config.config)
        merged_hyperparameters = {
            **{
                str(key): value
                for key, value in dict(
                    config_payload.get("hyperparameters") or {}
                ).items()
            },
            **dict(request.hyperparameters),
        }

        algorithm = str(config_payload.get("algorithm") or "").strip().lower()
        backend = _SENTIMENT_ALGORITHM_BACKENDS.get(algorithm)
        if backend is None:
            supported = ", ".join(sorted(_SENTIMENT_ALGORITHM_BACKENDS))
            raise UnsupportedLifecycleModelError(
                f"Pretrained activation is not implemented for algorithm "
                f"'{algorithm or 'unknown'}'. Supported pretrained sentiment "
                f"backends: {supported}"
            )

        try:
            sentiment_config = self._build_sentiment_config(
                backend=backend,
                hyperparameters=merged_hyperparameters,
            )
        except (ValueError, TypeError) as exc:
            raise LifecycleValidationError(
                f"Invalid pretrained activation hyperparameters: {exc}"
            ) from exc

        try:
            trainer = NewsSentimentTrainer(config=sentiment_config, model_id=model_id)
            self._smoke_check_sentiment(trainer)
        except Exception as exc:
            self._mark_error(model_id, str(exc))
            raise LifecycleOperationError(
                f"Pretrained activation failed for model '{model_id}': {exc}"
            ) from exc

        activation_config = {
            "hyperparameters": merged_hyperparameters,
            "sentiment_backend": backend.value,
        }
        self._registry.attach_trainer(
            model_id,
            trainer=trainer,
            model_path=None,
            config=activation_config,
        )
        self._promote_ready(model_id)

        entry = self._require_supporting_entry(model_id)
        self._broadcast_state(model_id)
        return self._build_response(entry)

    def load_artifact(
        self, model_id: str, request: LoadArtifactRequest
    ) -> SupportingModelLifecycleResponse:
        """Validate and load an external artifact, then mark the model READY.

        Args:
            model_id: Identifier of the supporting model.
            request: Artifact load request containing the artifact path.

        Returns:
            The updated lifecycle response (READY on success).

        Raises:
            ModelNotFoundError: If the model does not exist.
            UnsupportedLifecycleModelError: If the model is a core RL model.
            LifecycleValidationError: If the artifact path does not exist.
            LifecycleOperationError: If loading or the smoke check fails.
        """

        entry = self._require_supporting_entry(model_id)
        artifact_path = Path(request.model_path).expanduser()
        if not artifact_path.exists():
            raise LifecycleValidationError(
                f"Artifact path does not exist: {request.model_path}"
            )

        trainer_class = entry.config.trainer_class
        resolved_path = str(artifact_path)

        try:
            if trainer_class == _SB3_TRAINER_CLASS or entry.config.model_type == "rl":
                trainer = self._load_supporting_rl_artifact(
                    entry=entry, artifact_path=resolved_path
                )
                self._registry.attach_trainer(
                    model_id,
                    trainer=trainer,
                    model_path=resolved_path,
                )
            else:
                self._registry.load_model(model_id, model_path=resolved_path)
                loaded_entry = self._require_supporting_entry(model_id)
                self._smoke_check_loaded_entry(loaded_entry)
        except (LifecycleValidationError, UnsupportedLifecycleModelError):
            raise
        except RegistryError as exc:
            raise LifecycleOperationError(
                f"Failed to load artifact for model '{model_id}': {exc}"
            ) from exc
        except Exception as exc:
            self._mark_error(model_id, str(exc))
            raise LifecycleOperationError(
                f"Failed to load artifact for model '{model_id}': {exc}"
            ) from exc

        self._promote_ready(model_id)
        entry = self._require_supporting_entry(model_id)
        self._broadcast_state(model_id)
        return self._build_response(entry)

    def unload(self, model_id: str) -> SupportingModelLifecycleResponse:
        """Unload a supporting model trainer and transition to UNLOADED.

        Args:
            model_id: Identifier of the supporting model.

        Returns:
            The updated lifecycle response (UNLOADED on success).

        Raises:
            ModelNotFoundError: If the model does not exist.
            UnsupportedLifecycleModelError: If the model is a core RL model.
            LifecycleOperationError: If the unload transition is invalid.
        """

        self._require_supporting_entry(model_id)
        try:
            self._registry.unload_model(model_id)
        except RegistryError as exc:
            raise LifecycleOperationError(
                f"Failed to unload model '{model_id}': {exc}"
            ) from exc

        self._broadcast_state(model_id)
        entry = self._require_supporting_entry(model_id)
        return self._build_response(entry)

    def _require_supporting_entry(self, model_id: str) -> ModelEntry:
        """Return the supporting entry or raise a typed error.

        Raises:
            ModelNotFoundError: If no supporting model exists with this ID.
            UnsupportedLifecycleModelError: If the ID resolves to a core RL model.
        """

        entry = self._model_service.get_supporting_entry(model_id)
        if entry is not None:
            return entry

        # Distinguish "core model" from "unknown" for a clear 409 vs 404.
        try:
            model = self._model_service.get_model(model_id)
        except ModelNotFoundError:
            raise ModelNotFoundError(
                f"Supporting model '{model_id}' not found"
            ) from None

        if model.model_type == ModelType.CORE_RL:
            raise UnsupportedLifecycleModelError(
                f"Model '{model_id}' is a core RL model and has no supporting "
                "model lifecycle"
            )

        raise ModelNotFoundError(f"Supporting model '{model_id}' not found")

    def _build_sentiment_config(
        self,
        backend: SentimentModelType,
        hyperparameters: dict[str, int | float | str | bool],
    ) -> SentimentConfig:
        """Map hyperparameters into a concrete sentiment backend configuration."""

        kwargs: dict[str, object] = {"model_type": backend}
        if "model_name" in hyperparameters:
            kwargs["model_name"] = str(hyperparameters["model_name"])
        if "max_length" in hyperparameters:
            kwargs["max_length"] = int(hyperparameters["max_length"])
        if "device" in hyperparameters:
            kwargs["device"] = str(hyperparameters["device"])
        if "batch_size" in hyperparameters:
            kwargs["batch_size"] = int(hyperparameters["batch_size"])
        return SentimentConfig(**kwargs)  # type: ignore[arg-type]

    def _smoke_check_sentiment(self, trainer: NewsSentimentTrainer) -> None:
        """Run a minimal sentiment prediction to confirm the backend works."""

        prediction = trainer.predict(_SMOKE_TEXT)
        value = (
            prediction[0].value if isinstance(prediction, list) else prediction.value
        )
        if value is None:
            raise LifecycleOperationError(
                "Sentiment smoke prediction returned no value"
            )

    def _smoke_check_loaded_entry(self, entry: ModelEntry) -> None:
        """Run a minimal prediction against a freshly loaded ML trainer."""

        trainer = entry.trainer
        if not isinstance(trainer, NewsSentimentTrainer):
            raise LifecycleOperationError(
                "Loaded supporting ML trainer is not a NewsSentimentTrainer"
            )
        self._smoke_check_sentiment(trainer)

    def _load_supporting_rl_artifact(
        self, entry: ModelEntry, artifact_path: str
    ) -> StableBaselines3Trainer:
        """Instantiate a configured SB3 trainer and load the artifact zip."""

        algorithm_value = str(entry.config.config.get("algorithm") or "ppo")
        try:
            algorithm = SB3Algorithm(algorithm_value.strip().lower())
        except ValueError as exc:
            supported = ", ".join(sorted(item.value for item in SB3Algorithm))
            raise LifecycleValidationError(
                f"Unsupported supporting RL algorithm '{algorithm_value}'. "
                f"Supported algorithms: {supported}"
            ) from exc

        trainer = StableBaselines3Trainer(algorithm=algorithm)
        trainer.load(artifact_path)
        if not trainer.is_trained:
            raise LifecycleOperationError(
                "Supporting RL artifact loaded but trainer reports not trained"
            )
        return trainer

    def _promote_ready(self, model_id: str) -> None:
        """Transition a LOADED model to READY."""

        try:
            self._registry.set_state(model_id, ModelState.READY)
        except RegistryError as exc:
            raise LifecycleOperationError(
                f"Failed to mark model '{model_id}' READY: {exc}"
            ) from exc

    def _mark_error(self, model_id: str, message: str) -> None:
        """Best-effort transition to ERROR, ignoring invalid-transition noise."""

        try:
            self._registry.set_state(model_id, ModelState.ERROR, error=message)
        except RegistryError:
            logger.debug(
                "Could not transition model '%s' to ERROR after failure", model_id
            )

    def _broadcast_state(self, model_id: str) -> None:
        """Broadcast updated model state when a websocket broadcaster is wired."""

        broadcast = getattr(self._model_service, "broadcast_model_state", None)
        if callable(broadcast):
            try:
                broadcast(model_id)
            except Exception as exc:  # pragma: no cover - defensive only
                logger.debug("Model state broadcast failed for '%s': %s", model_id, exc)

    def _build_response(self, entry: ModelEntry) -> SupportingModelLifecycleResponse:
        """Build a lifecycle response with readiness checks for an entry."""

        config_payload = dict(entry.config.config)
        algorithm = (
            str(config_payload["algorithm"])
            if config_payload.get("algorithm") is not None
            else None
        )
        model_type = (
            ModelType.SUPPORTING_ML.value
            if entry.config.model_type == "ml"
            else ModelType.SUPPORTING_RL.value
        )

        has_trainer = entry.trainer is not None
        is_loaded = entry.state in {ModelState.LOADED, ModelState.READY}
        has_artifact_or_pretrained = (
            entry.config.model_path is not None
            or config_payload.get("sentiment_backend") is not None
        )
        no_error = entry.error_message is None

        readiness_checks = [
            ReadinessCheck(
                name="trainer_loaded",
                passed=has_trainer,
                detail=(
                    "Trainer instance attached"
                    if has_trainer
                    else "No trainer instance loaded"
                ),
            ),
            ReadinessCheck(
                name="backend_resolved",
                passed=has_artifact_or_pretrained,
                detail=(
                    "Artifact path or pretrained backend configured"
                    if has_artifact_or_pretrained
                    else "No artifact path or pretrained backend configured"
                ),
            ),
            ReadinessCheck(
                name="loaded_state",
                passed=is_loaded,
                detail=f"Current state is {entry.state.value}",
            ),
            ReadinessCheck(
                name="no_error",
                passed=no_error,
                detail=entry.error_message or "No lifecycle error recorded",
            ),
        ]

        return SupportingModelLifecycleResponse(
            model_id=entry.config.model_id,
            model_type=model_type,
            state=entry.state.value,
            model_path=entry.config.model_path,
            trainer_class=entry.config.trainer_class,
            input_data_types=[item.value for item in entry.config.input_data_types],
            signal_type=entry.config.signal_type.value,
            algorithm=algorithm,
            last_error=entry.error_message,
            is_ready=entry.state == ModelState.READY,
            readiness_checks=readiness_checks,
        )


def get_supporting_lifecycle_service(
    model_service: ModelService,
) -> SupportingModelLifecycleService:
    """Build a lifecycle service bound to the shared model service."""

    return SupportingModelLifecycleService(model_service=model_service)


__all__ = [
    "SupportingModelLifecycleService",
    "SupportingLifecycleError",
    "UnsupportedLifecycleModelError",
    "LifecycleValidationError",
    "LifecycleOperationError",
    "get_supporting_lifecycle_service",
]
