"""Business logic for model, strategy, and generation management endpoints."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from math import ceil
from pathlib import Path
from threading import RLock
from typing import Collection, TypeVar
from uuid import uuid4

from algotrading.api.schemas.common import PaginatedResponse
from algotrading.api.schemas.generations import (
    GenerationComparison,
    GenerationDetail,
    GenerationSummary,
)
from algotrading.api.schemas.models import (
    ModelConfigCreate,
    ModelConfigResponse,
    ModelConfigUpdate,
    ModelType,
)
from algotrading.api.schemas.strategies import StrategyDetail, StrategyInfo
from algotrading.src.data_pipeline import DataFrequency, DataType
from algotrading.src.models.registry import (
    CustomStrategyRegistry,
    ModelAlreadyExistsError,
    ModelEntry,
    ModelEntryConfig,
)
from algotrading.src.models.registry import (
    ModelNotFoundError as RegistryModelNotFoundError,
)
from algotrading.src.models.registry import (
    ModelState,
    SupportingModelRegistry,
)
from algotrading.src.models.signals import SignalType
from algotrading.src.models.tracking import (
    EvaluationMetrics,
    Generation,
    GenerationTracker,
    JsonFileStorage,
)
from fastapi import Request

logger = logging.getLogger(__name__)


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

_SUPPORTED_TRAINER_ALGORITHMS: dict[str, set[str]] = {
    "stable_baselines3": {"ppo", "dqn", "a2c"},
    "sklearn": {"random_forest", "gradient_boosting", "logistic_regression"},
    "tensorflow": {"lstm", "mlp"},
    "huggingface": {"transformer_sentiment", "finbert", "vader", "provider"},
}

_HYPERPARAMETER_SCHEMAS: dict[
    tuple[str, str], dict[str, type[int | float | str | bool]]
] = {
    ("stable_baselines3", "ppo"): {
        "learning_rate": float,
        "n_steps": int,
        "batch_size": int,
        "n_epochs": int,
        "gamma": float,
        "clip_range": float,
    },
    ("stable_baselines3", "dqn"): {
        "learning_rate": float,
        "buffer_size": int,
        "batch_size": int,
        "gamma": float,
        "exploration_fraction": float,
        "target_update_interval": int,
    },
    ("sklearn", "random_forest"): {
        "n_estimators": int,
        "max_depth": int,
    },
    ("tensorflow", "lstm"): {
        "units": int,
        "num_layers": int,
        "dropout": float,
    },
    ("huggingface", "transformer_sentiment"): {
        "model_name": str,
        "max_length": int,
    },
}

_TRAINER_CLASS_BY_MODEL_TYPE: dict[ModelType, str] = {
    ModelType.SUPPORTING_ML: "algotrading.src.models.supporting.sentiment.news_sentiment.NewsSentimentTrainer",
    ModelType.SUPPORTING_RL: "algotrading.src.trainers.sb3_trainer.StableBaselines3Trainer",
}

_DATA_TYPE_ALIASES: dict[str, DataType] = {
    "market_bar": DataType.MARKET_BAR,
    "market_data": DataType.MARKET_BAR,
    "news_text": DataType.NEWS_TEXT,
    "indicator": DataType.INDICATOR,
    "technical_indicator": DataType.INDICATOR,
    "signal": DataType.SIGNAL,
    "model_signal": DataType.SIGNAL,
    "supporting_model_signal": DataType.SIGNAL,
}

_FREQUENCY_ALIASES: dict[str, DataFrequency] = {
    "tick": DataFrequency.TICK,
    "realtime": DataFrequency.TICK,
    "1s": DataFrequency.SECOND_1,
    "5s": DataFrequency.SECOND_5,
    "10s": DataFrequency.SECOND_10,
    "30s": DataFrequency.SECOND_30,
    "1m": DataFrequency.MINUTE_1,
    "5m": DataFrequency.MINUTE_5,
    "15m": DataFrequency.MINUTE_15,
    "1h": DataFrequency.HOUR_1,
    "1d": DataFrequency.DAY_1,
    "same as market data": DataFrequency.IRREGULAR,
}

_NUMERIC_TYPES: tuple[type[object], ...] = (int, float)
ResponseItemT = TypeVar("ResponseItemT")


class ModelServiceError(Exception):
    """Base error class for model service operations."""


class ModelValidationError(ModelServiceError):
    """Raised when a model configuration is invalid."""


class ModelNotFoundError(ModelServiceError):
    """Raised when a model does not exist."""


class InvalidModelStateError(ModelServiceError):
    """Raised when an operation is invalid for a model's current state."""


class StrategyNotFoundError(ModelServiceError):
    """Raised when a requested strategy cannot be found."""


class GenerationNotFoundError(ModelServiceError):
    """Raised when a requested generation cannot be found."""


@dataclass(slots=True)
class CoreModelRecord:
    """Persisted representation of a core RL model configuration."""

    model_id: str
    name: str
    description: str | None
    trainer_type: str
    algorithm: str
    hyperparameters: dict[str, int | float | str | bool]
    training_data_config: dict[str, object]
    supporting_model_ids: list[str]
    strategy_ids: list[str]
    environment_config: dict[str, object]
    reward_function: str
    created_at: datetime
    updated_at: datetime
    state: str = "configured"

    def to_dict(self) -> dict[str, object]:
        """Serialize core model data for JSON persistence."""

        return {
            "model_id": self.model_id,
            "name": self.name,
            "description": self.description,
            "trainer_type": self.trainer_type,
            "algorithm": self.algorithm,
            "hyperparameters": dict(self.hyperparameters),
            "training_data_config": dict(self.training_data_config),
            "supporting_model_ids": list(self.supporting_model_ids),
            "strategy_ids": list(self.strategy_ids),
            "environment_config": dict(self.environment_config),
            "reward_function": self.reward_function,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> CoreModelRecord:
        """Deserialize a persisted core model record."""

        return cls(
            model_id=str(payload["model_id"]),
            name=str(payload["name"]),
            description=(
                str(payload["description"])
                if payload.get("description") is not None
                else None
            ),
            trainer_type=str(payload["trainer_type"]),
            algorithm=str(payload["algorithm"]),
            hyperparameters={
                str(key): value
                for key, value in dict(payload.get("hyperparameters") or {}).items()
                if isinstance(value, (int, float, str, bool))
            },
            training_data_config={
                str(key): value
                for key, value in dict(
                    payload.get("training_data_config") or {}
                ).items()
            },
            supporting_model_ids=[
                str(value) for value in list(payload.get("supporting_model_ids") or [])
            ],
            strategy_ids=[
                str(value) for value in list(payload.get("strategy_ids") or [])
            ],
            environment_config={
                str(key): value
                for key, value in dict(payload.get("environment_config") or {}).items()
            },
            reward_function=str(payload["reward_function"]),
            created_at=_parse_datetime(str(payload["created_at"])),
            updated_at=_parse_datetime(str(payload["updated_at"])),
            state=str(payload.get("state") or "configured"),
        )


class ModelService:
    """Service layer exposing model-management business logic.

    The service is intentionally stateful for API process-local workflows.
    Supporting model configurations are delegated to the existing supporting
    registry. Core RL model configurations are persisted in a lightweight JSON
    store to preserve state across API restarts.
    """

    def __init__(
        self,
        supporting_registry: SupportingModelRegistry,
        strategy_registry: CustomStrategyRegistry,
        generation_tracker: GenerationTracker,
        core_models_path: str | Path | None = None,
    ) -> None:
        """Initialize service dependencies.

        Args:
            supporting_registry: Registry used for supporting ML/RL model entries.
            strategy_registry: Registry used for strategy discovery and lookup.
            generation_tracker: Tracker used for generation history APIs.
            core_models_path: Optional persistence location for core model configs.
        """

        self.supporting_registry = supporting_registry
        self.strategy_registry = strategy_registry
        self.generation_tracker = generation_tracker
        self._core_models_path = Path(core_models_path) if core_models_path else None
        self._core_models: dict[str, CoreModelRecord] = {}
        self._lock = RLock()

        self._load_core_models()

    def create_model(self, config: ModelConfigCreate) -> ModelConfigResponse:
        """Create a core or supporting model configuration."""

        self._validate_config(config)

        if config.model_type == ModelType.CORE_RL:
            self._validate_supporting_model_ids(config.supporting_model_ids)
            self._validate_strategy_ids(config.strategy_ids)
            return self._create_core_model(config)

        self._validate_no_supporting_inputs(config.input_data_types)
        return self._create_supporting_model(config)

    def get_supporting_entry(self, model_id: str) -> ModelEntry | None:
        """Return the raw supporting registry entry for a model, if present.

        This exposes the underlying registry entry (including lifecycle state,
        trainer instance, and artifact path) for lifecycle operations without
        coercing it through the API response schema.

        Args:
            model_id: Identifier of the supporting model.

        Returns:
            The supporting :class:`ModelEntry`, or ``None`` if the ID is not a
            registered supporting model.
        """

        return self.supporting_registry.get(model_id)

    def get_model(self, model_id: str) -> ModelConfigResponse:
        """Return one model by ID from either core or supporting stores."""

        with self._lock:
            core_record = self._core_models.get(model_id)
            if core_record is not None:
                return self._core_record_to_response(core_record)

        entry = self.supporting_registry.get(model_id)
        if entry is not None:
            return self._supporting_entry_to_response(entry)

        raise ModelNotFoundError(f"Model '{model_id}' not found")

    def update_model(
        self, model_id: str, updates: ModelConfigUpdate
    ) -> ModelConfigResponse:
        """Update mutable model configuration fields."""

        if updates.model_type is not None:
            existing_type = self.get_model(model_id).model_type
            if updates.model_type != existing_type:
                raise ModelValidationError(
                    "model_type cannot be changed for an existing model"
                )

        with self._lock:
            core_record = self._core_models.get(model_id)

        if core_record is not None:
            return self._update_core_model(core_record, updates)

        entry = self.supporting_registry.get(model_id)
        if entry is not None:
            return self._update_supporting_model(entry, updates)

        raise ModelNotFoundError(f"Model '{model_id}' not found")

    def delete_model(self, model_id: str) -> None:
        """Delete a model configuration."""

        with self._lock:
            if model_id in self._core_models:
                self._core_models.pop(model_id)
                self._persist_core_models()
                return

        removed = self.supporting_registry.unregister(model_id)
        if not removed:
            raise ModelNotFoundError(f"Model '{model_id}' not found")

    def list_models(
        self,
        model_type: ModelType | None = None,
        page: int = 1,
        page_size: int = 20,
        model_types: Collection[ModelType] | None = None,
    ) -> PaginatedResponse[ModelConfigResponse]:
        """List models with optional type filtering and pagination."""

        selected_types = set(
            model_types or ([] if model_type is None else [model_type])
        )

        models: list[ModelConfigResponse] = []

        with self._lock:
            core_values = list(self._core_models.values())
        for record in core_values:
            if selected_types and ModelType.CORE_RL not in selected_types:
                continue
            models.append(self._core_record_to_response(record))

        for entry in self.supporting_registry.get_all():
            response = self._supporting_entry_to_response(entry)
            if selected_types and response.model_type not in selected_types:
                continue
            models.append(response)

        models.sort(key=lambda item: item.updated_at, reverse=True)
        return _paginate(items=models, page=page, page_size=page_size)

    def get_generations(
        self,
        model_id: str,
        page: int = 1,
        page_size: int = 20,
    ) -> PaginatedResponse[GenerationSummary]:
        """Return paginated generation summaries for a model."""

        self.get_model(model_id)
        generations = self.generation_tracker.get_generations_for_model(model_id)
        summaries = [self._to_generation_summary(item) for item in generations]
        summaries.sort(key=lambda item: item.generation_number, reverse=True)
        return _paginate(items=summaries, page=page, page_size=page_size)

    def get_generation_detail(
        self, model_id: str, generation_id: str
    ) -> GenerationDetail:
        """Return detailed generation information for one model generation."""

        self.get_model(model_id)
        generation = self.generation_tracker.get_generation(generation_id)
        if generation is None or generation.model_id != model_id:
            raise GenerationNotFoundError(
                f"Generation '{generation_id}' not found for model '{model_id}'"
            )
        return self._to_generation_detail(generation)

    def get_generation_detail_by_id(self, generation_id: str) -> GenerationDetail:
        """Return detailed generation information using a generation ID only."""

        generation = self.generation_tracker.get_generation(generation_id)
        if generation is None:
            raise GenerationNotFoundError(f"Generation '{generation_id}' not found")
        return self._to_generation_detail(generation)

    def compare_generations(
        self,
        model_id: str,
        generation_ids: list[str],
    ) -> GenerationComparison:
        """Compare generations for the same model and compute metric deltas."""

        self.get_model(model_id)

        if len(generation_ids) < 2:
            raise ModelValidationError("At least two generation_ids are required")

        unique_ids = list(dict.fromkeys(generation_ids))
        generations: list[Generation] = []
        for generation_id in unique_ids:
            generation = self.generation_tracker.get_generation(generation_id)
            if generation is None or generation.model_id != model_id:
                raise GenerationNotFoundError(
                    f"Generation '{generation_id}' not found for model '{model_id}'"
                )
            generations.append(generation)

        summaries = [self._to_generation_summary(item) for item in generations]
        summaries.sort(key=lambda item: item.generation_number)
        deltas = self._build_metric_deltas(summaries)

        return GenerationComparison(generations=summaries, metric_deltas=deltas)

    def list_strategies(
        self, signal_type: SignalType | None = None
    ) -> list[StrategyInfo]:
        """Return all registered strategies with optional signal-type filter."""

        entries = self.strategy_registry.get_all()
        if signal_type is not None:
            entries = [entry for entry in entries if entry.signal_type == signal_type]

        strategies = [
            StrategyInfo(
                strategy_id=entry.strategy_id,
                name=entry.name,
                signal_type=entry.signal_type,
                description=entry.description,
                version=entry.version,
                state="ready",
            )
            for entry in entries
        ]
        strategies.sort(key=lambda item: item.name.lower())
        return strategies

    def get_strategy(self, strategy_id: str) -> StrategyDetail:
        """Return details for one strategy."""

        entry = self.strategy_registry.get(strategy_id)
        if entry is None:
            raise StrategyNotFoundError(f"Strategy '{strategy_id}' not found")

        return StrategyDetail(
            strategy_id=entry.strategy_id,
            name=entry.name,
            signal_type=entry.signal_type,
            description=entry.description,
            version=entry.version,
            state="ready",
            filepath=str(entry.filepath),
            class_name=entry.strategy_class.__name__,
        )

    def _create_core_model(self, config: ModelConfigCreate) -> ModelConfigResponse:
        """Create a persisted core RL model record."""

        now = datetime.now(tz=UTC)
        model_id = f"core-rl-{uuid4().hex[:8]}"

        record = CoreModelRecord(
            model_id=model_id,
            name=config.name,
            description=config.description,
            trainer_type=config.trainer_type,
            algorithm=config.algorithm,
            hyperparameters=dict(config.hyperparameters),
            training_data_config=dict(config.training_data_config),
            supporting_model_ids=list(config.supporting_model_ids),
            strategy_ids=list(config.strategy_ids),
            environment_config=dict(config.environment_config),
            reward_function=config.reward_function or "",
            created_at=now,
            updated_at=now,
            state="configured",
        )

        with self._lock:
            self._core_models[model_id] = record
            self._persist_core_models()

        return self._core_record_to_response(record)

    def _create_supporting_model(
        self, config: ModelConfigCreate
    ) -> ModelConfigResponse:
        """Create a supporting model via the existing supporting registry."""

        model_id = (
            f"supporting-ml-{uuid4().hex[:8]}"
            if config.model_type == ModelType.SUPPORTING_ML
            else f"supporting-rl-{uuid4().hex[:8]}"
        )

        input_types = [
            self._parse_input_data_type(value) for value in config.input_data_types
        ]
        input_frequency = self._parse_frequency(config.input_frequency or "")

        now_iso = datetime.now(tz=UTC).isoformat()
        payload = {
            "name": config.name,
            "description": config.description,
            "trainer_type": config.trainer_type,
            "algorithm": config.algorithm,
            "hyperparameters": dict(config.hyperparameters),
            "training_data_config": dict(config.training_data_config),
            "input_data_types": list(config.input_data_types),
            "input_frequency": config.input_frequency,
            "created_at": now_iso,
            "updated_at": now_iso,
        }

        registry_config = ModelEntryConfig(
            model_id=model_id,
            model_type=("ml" if config.model_type == ModelType.SUPPORTING_ML else "rl"),
            signal_type=config.signal_type or SignalType.CUSTOM,
            trainer_class=self._resolve_trainer_class(
                model_type=config.model_type,
                trainer_type=config.trainer_type,
            ),
            input_data_types=input_types,
            input_frequency=input_frequency,
            description=config.description,
            config=payload,
        )

        try:
            self.supporting_registry.register(registry_config)
        except ModelAlreadyExistsError as exc:
            raise ModelValidationError(str(exc)) from exc

        created = self.supporting_registry.get(model_id)
        if created is None:
            raise ModelServiceError(
                f"Supporting model '{model_id}' was not found after registration"
            )
        return self._supporting_entry_to_response(created)

    def _update_core_model(
        self,
        core_record: CoreModelRecord,
        updates: ModelConfigUpdate,
    ) -> ModelConfigResponse:
        """Update a core model record using partial updates."""

        merged_payload = {
            "name": updates.name if updates.name is not None else core_record.name,
            "description": (
                updates.description
                if updates.description is not None
                else core_record.description
            ),
            "model_type": ModelType.CORE_RL,
            "signal_type": None,
            "trainer_type": (
                updates.trainer_type
                if updates.trainer_type is not None
                else core_record.trainer_type
            ),
            "algorithm": (
                updates.algorithm
                if updates.algorithm is not None
                else core_record.algorithm
            ),
            "hyperparameters": (
                dict(updates.hyperparameters)
                if updates.hyperparameters is not None
                else dict(core_record.hyperparameters)
            ),
            "training_data_config": (
                dict(updates.training_data_config)
                if updates.training_data_config is not None
                else dict(core_record.training_data_config)
            ),
            "supporting_model_ids": (
                list(updates.supporting_model_ids)
                if updates.supporting_model_ids is not None
                else list(core_record.supporting_model_ids)
            ),
            "strategy_ids": (
                list(updates.strategy_ids)
                if updates.strategy_ids is not None
                else list(core_record.strategy_ids)
            ),
            "environment_config": (
                dict(updates.environment_config)
                if updates.environment_config is not None
                else dict(core_record.environment_config)
            ),
            "reward_function": (
                updates.reward_function
                if updates.reward_function is not None
                else core_record.reward_function
            ),
            "input_data_types": [],
            "input_frequency": None,
        }

        candidate = ModelConfigCreate(**merged_payload)
        self._validate_config(candidate)
        self._validate_supporting_model_ids(candidate.supporting_model_ids)
        self._validate_strategy_ids(candidate.strategy_ids)

        updated = CoreModelRecord(
            model_id=core_record.model_id,
            name=candidate.name,
            description=candidate.description,
            trainer_type=candidate.trainer_type,
            algorithm=candidate.algorithm,
            hyperparameters=dict(candidate.hyperparameters),
            training_data_config=dict(candidate.training_data_config),
            supporting_model_ids=list(candidate.supporting_model_ids),
            strategy_ids=list(candidate.strategy_ids),
            environment_config=dict(candidate.environment_config),
            reward_function=candidate.reward_function or "",
            created_at=core_record.created_at,
            updated_at=datetime.now(tz=UTC),
            state=core_record.state,
        )

        with self._lock:
            self._core_models[core_record.model_id] = updated
            self._persist_core_models()

        return self._core_record_to_response(updated)

    def _update_supporting_model(
        self,
        entry: ModelEntry,
        updates: ModelConfigUpdate,
    ) -> ModelConfigResponse:
        """Update supporting model configuration through registry updates."""

        payload = self._supporting_entry_to_create_payload(entry)

        if updates.name is not None:
            payload["name"] = updates.name
        if updates.description is not None:
            payload["description"] = updates.description
        if updates.trainer_type is not None:
            payload["trainer_type"] = updates.trainer_type
        if updates.algorithm is not None:
            payload["algorithm"] = updates.algorithm
        if updates.hyperparameters is not None:
            payload["hyperparameters"] = dict(updates.hyperparameters)
        if updates.training_data_config is not None:
            payload["training_data_config"] = dict(updates.training_data_config)
        if updates.signal_type is not None:
            payload["signal_type"] = updates.signal_type
        if updates.input_data_types is not None:
            payload["input_data_types"] = list(updates.input_data_types)
        if updates.input_frequency is not None:
            payload["input_frequency"] = updates.input_frequency

        candidate = ModelConfigCreate(**payload)
        self._validate_config(candidate)
        self._validate_no_supporting_inputs(candidate.input_data_types)

        now_iso = datetime.now(tz=UTC).isoformat()
        config_payload = {
            "name": candidate.name,
            "description": candidate.description,
            "trainer_type": candidate.trainer_type,
            "algorithm": candidate.algorithm,
            "hyperparameters": dict(candidate.hyperparameters),
            "training_data_config": dict(candidate.training_data_config),
            "input_data_types": list(candidate.input_data_types),
            "input_frequency": candidate.input_frequency,
            "created_at": payload.get("created_at", now_iso),
            "updated_at": now_iso,
        }

        registry_updates: dict[str, object] = {
            "model_type": (
                "ml" if candidate.model_type == ModelType.SUPPORTING_ML else "rl"
            ),
            "signal_type": candidate.signal_type,
            "trainer_class": self._resolve_trainer_class(
                model_type=candidate.model_type,
                trainer_type=candidate.trainer_type,
            ),
            "input_data_types": [
                self._parse_input_data_type(value)
                for value in candidate.input_data_types
            ],
            "input_frequency": self._parse_frequency(candidate.input_frequency or ""),
            "description": candidate.description,
            "config": config_payload,
        }

        try:
            self.supporting_registry.update_config(
                entry.config.model_id, registry_updates
            )
        except RegistryModelNotFoundError as exc:
            raise ModelNotFoundError(str(exc)) from exc

        updated_entry = self.supporting_registry.get(entry.config.model_id)
        if updated_entry is None:
            raise ModelNotFoundError(
                f"Model '{entry.config.model_id}' not found after update"
            )
        return self._supporting_entry_to_response(updated_entry)

    def _validate_config(self, config: ModelConfigCreate) -> None:
        """Validate cross-field business rules for model creation/update."""

        trainer_key = config.trainer_type.strip().lower()
        algorithm_key = config.algorithm.strip().lower()

        if trainer_key not in _SUPPORTED_TRAINER_ALGORITHMS:
            supported = ", ".join(sorted(_SUPPORTED_TRAINER_ALGORITHMS.keys()))
            raise ModelValidationError(
                f"Unsupported trainer_type '{config.trainer_type}'. Supported values: {supported}"
            )

        if algorithm_key not in _SUPPORTED_TRAINER_ALGORITHMS[trainer_key]:
            supported_algorithms = ", ".join(
                sorted(_SUPPORTED_TRAINER_ALGORITHMS[trainer_key])
            )
            raise ModelValidationError(
                f"Unsupported algorithm '{config.algorithm}' for trainer '{config.trainer_type}'. "
                f"Supported algorithms: {supported_algorithms}"
            )

        schema = _HYPERPARAMETER_SCHEMAS.get((trainer_key, algorithm_key), {})
        for key, value in config.hyperparameters.items():
            if not isinstance(value, (int, float, str, bool)):
                raise ModelValidationError(
                    f"Hyperparameter '{key}' has unsupported value type '{type(value).__name__}'"
                )
            expected_type = schema.get(key)
            if expected_type is None:
                continue
            if not _matches_type(value, expected_type):
                raise ModelValidationError(
                    f"Hyperparameter '{key}' must be of type '{expected_type.__name__}'"
                )

        if config.model_type == ModelType.CORE_RL:
            if not config.reward_function:
                raise ModelValidationError(
                    "reward_function is required for core RL models"
                )
            self._validate_unique_ids(
                config.supporting_model_ids, field_name="supporting_model_ids"
            )
            self._validate_unique_ids(config.strategy_ids, field_name="strategy_ids")
            return

        if config.signal_type is None:
            raise ModelValidationError("signal_type is required for supporting models")

        if not config.input_data_types:
            raise ModelValidationError(
                "input_data_types is required for supporting models"
            )

        if not config.input_frequency:
            raise ModelValidationError(
                "input_frequency is required for supporting models"
            )

    def _validate_no_supporting_inputs(self, input_data_types: list[str]) -> None:
        """Validate supporting model input dependency constraints."""

        for value in input_data_types:
            normalized = value.strip().lower()
            alias = _DATA_TYPE_ALIASES.get(normalized)
            if alias == DataType.SIGNAL:
                raise ModelValidationError(
                    "Supporting models cannot depend on supporting model signal inputs"
                )

    def _validate_supporting_model_ids(self, ids: list[str]) -> None:
        """Ensure all referenced supporting models exist and are in READY state."""

        for model_id in ids:
            entry = self.supporting_registry.get(model_id)
            if entry is None:
                raise ModelValidationError(
                    f"supporting_model_id '{model_id}' does not exist"
                )
            if entry.state != ModelState.READY:
                raise InvalidModelStateError(
                    f"supporting_model_id '{model_id}' is not READY (current state: {entry.state.value})"
                )

    def _validate_strategy_ids(self, ids: list[str]) -> None:
        """Ensure all referenced strategies exist in the strategy registry."""

        for strategy_id in ids:
            if self.strategy_registry.get(strategy_id) is None:
                raise ModelValidationError(
                    f"strategy_id '{strategy_id}' does not exist"
                )

    def _validate_unique_ids(self, ids: list[str], field_name: str) -> None:
        """Validate list values contain no duplicates."""

        if len(ids) != len(set(ids)):
            raise ModelValidationError(f"{field_name} contains duplicate identifiers")

    def _resolve_trainer_class(self, model_type: ModelType, trainer_type: str) -> str:
        """Resolve a trainer class path for supporting model registry entries."""

        if model_type in _TRAINER_CLASS_BY_MODEL_TYPE:
            return _TRAINER_CLASS_BY_MODEL_TYPE[model_type]

        trainer_key = trainer_type.strip().lower()
        if trainer_key == "stable_baselines3":
            return "algotrading.src.trainers.sb3_trainer.StableBaselines3Trainer"
        if trainer_key == "sklearn":
            return "algotrading.src.models.supporting.sentiment.news_sentiment.NewsSentimentTrainer"
        if trainer_key == "tensorflow":
            return "algotrading.src.models.supporting.sentiment.news_sentiment.NewsSentimentTrainer"
        if trainer_key == "huggingface":
            return "algotrading.src.models.supporting.sentiment.news_sentiment.NewsSentimentTrainer"

        return "algotrading.src.models.supporting.sentiment.news_sentiment.NewsSentimentTrainer"

    def _parse_input_data_type(self, value: str) -> DataType:
        """Normalize API input-data type string to DataType enum."""

        normalized = value.strip().lower()
        aliased = _DATA_TYPE_ALIASES.get(normalized)
        if aliased is not None:
            return aliased

        try:
            return DataType(value)
        except ValueError:
            pass

        try:
            return DataType(value.strip().upper())
        except ValueError as exc:
            raise ModelValidationError(
                f"Unsupported input_data_type '{value}'"
            ) from exc

    def _parse_frequency(self, value: str) -> DataFrequency:
        """Normalize API frequency string to DataFrequency enum."""

        normalized = value.strip().lower()
        aliased = _FREQUENCY_ALIASES.get(normalized)
        if aliased is not None:
            return aliased

        try:
            return DataFrequency(value)
        except ValueError:
            pass

        upper = value.strip().upper()
        if upper in DataFrequency.__members__:
            return DataFrequency[upper]

        raise ModelValidationError(f"Unsupported input_frequency '{value}'")

    def _core_record_to_response(self, record: CoreModelRecord) -> ModelConfigResponse:
        """Convert internal core model record to API response schema."""

        return ModelConfigResponse(
            model_id=record.model_id,
            name=record.name,
            description=record.description,
            model_type=ModelType.CORE_RL,
            signal_type=None,
            trainer_type=record.trainer_type,
            algorithm=record.algorithm,
            hyperparameters=dict(record.hyperparameters),
            training_data_config=dict(record.training_data_config),
            supporting_model_ids=list(record.supporting_model_ids),
            strategy_ids=list(record.strategy_ids),
            environment_config=dict(record.environment_config),
            reward_function=record.reward_function,
            input_data_types=[],
            input_frequency=None,
            created_at=record.created_at,
            updated_at=record.updated_at,
            state=record.state,
        )

    def _supporting_entry_to_response(self, entry: ModelEntry) -> ModelConfigResponse:
        """Convert supporting registry entry to API response schema."""

        payload = dict(entry.config.config)
        model_type = (
            ModelType.SUPPORTING_ML
            if entry.config.model_type == "ml"
            else ModelType.SUPPORTING_RL
        )

        created_at = _parse_datetime(
            str(payload.get("created_at") or entry.registered_at.isoformat())
        )
        updated_at = _parse_datetime(
            str(
                payload.get("updated_at")
                or (
                    entry.loaded_at.isoformat()
                    if entry.loaded_at
                    else entry.registered_at.isoformat()
                )
            )
        )

        return ModelConfigResponse(
            model_id=entry.config.model_id,
            name=str(payload.get("name") or entry.config.model_id),
            description=(
                str(payload["description"])
                if payload.get("description") is not None
                else entry.config.description
            ),
            model_type=model_type,
            signal_type=entry.config.signal_type,
            trainer_type=str(payload.get("trainer_type") or "unknown"),
            algorithm=str(payload.get("algorithm") or "unknown"),
            hyperparameters={
                str(key): value
                for key, value in dict(payload.get("hyperparameters") or {}).items()
                if isinstance(value, (int, float, str, bool))
            },
            training_data_config={
                str(key): value
                for key, value in dict(
                    payload.get("training_data_config") or {}
                ).items()
            },
            supporting_model_ids=[],
            strategy_ids=[],
            environment_config={},
            reward_function=None,
            input_data_types=[
                str(item) for item in list(payload.get("input_data_types") or [])
            ],
            input_frequency=(
                str(payload["input_frequency"])
                if payload.get("input_frequency") is not None
                else entry.config.input_frequency.value
            ),
            created_at=created_at,
            updated_at=updated_at,
            state=entry.state.value,
        )

    def _supporting_entry_to_create_payload(
        self, entry: ModelEntry
    ) -> dict[str, object]:
        """Convert supporting entry into ModelConfigCreate-compatible payload."""

        response = self._supporting_entry_to_response(entry)
        return {
            "name": response.name,
            "description": response.description,
            "model_type": response.model_type,
            "signal_type": response.signal_type,
            "trainer_type": response.trainer_type,
            "algorithm": response.algorithm,
            "hyperparameters": dict(response.hyperparameters),
            "training_data_config": dict(response.training_data_config),
            "supporting_model_ids": [],
            "strategy_ids": [],
            "environment_config": {},
            "reward_function": None,
            "input_data_types": list(response.input_data_types),
            "input_frequency": response.input_frequency,
            "created_at": response.created_at.isoformat(),
            "updated_at": response.updated_at.isoformat(),
        }

    def _to_generation_summary(self, generation: Generation) -> GenerationSummary:
        """Map domain generation to API summary schema."""

        metrics: dict[str, float | int | str | bool | None] = {}
        training_duration = 0.0
        final_reward: float | None = None

        if generation.training_metrics is not None:
            final_reward = generation.training_metrics.final_reward
            training_duration = generation.training_metrics.training_time_seconds
            metrics["final_reward"] = generation.training_metrics.final_reward
            metrics["mean_reward"] = generation.training_metrics.mean_reward
            metrics["std_reward"] = generation.training_metrics.std_reward
            metrics["episodes_completed"] = (
                generation.training_metrics.episodes_completed
            )
            metrics["timesteps_trained"] = generation.training_metrics.timesteps_trained
            if generation.training_metrics.custom_metrics:
                metrics.update(generation.training_metrics.custom_metrics)

        if generation.evaluation_metrics is not None:
            metrics.update(_evaluation_metrics_to_dict(generation.evaluation_metrics))

        return GenerationSummary(
            generation_id=generation.generation_id,
            generation_number=generation.generation_number,
            model_id=generation.model_id,
            created_at=generation.created_at,
            training_duration_seconds=training_duration,
            final_reward=final_reward,
            metrics=metrics,
        )

    def _to_generation_detail(self, generation: Generation) -> GenerationDetail:
        """Map domain generation to API detail schema."""

        summary = self._to_generation_summary(generation)
        training_config = {
            "status": generation.status,
            "tags": list(generation.tags),
            "notes": generation.notes,
            "parent_generation_id": generation.parent_generation_id,
        }

        return GenerationDetail(
            **summary.model_dump(),
            hyperparameters=dict(generation.hyperparameters),
            training_config=training_config,
            metrics_history=[],
            model_path=generation.model_path,
        )

    def _build_metric_deltas(
        self, summaries: list[GenerationSummary]
    ) -> dict[str, dict[str, float | None]]:
        """Build per-metric deltas using the first generation as baseline."""

        if not summaries:
            return {}

        baseline = summaries[0]
        baseline_metrics = baseline.metrics

        deltas: dict[str, dict[str, float | None]] = {}
        metric_names = sorted(
            {
                *baseline_metrics.keys(),
                *{name for summary in summaries[1:] for name in summary.metrics.keys()},
            }
        )

        for metric_name in metric_names:
            baseline_value = baseline_metrics.get(metric_name)
            metric_delta: dict[str, float | None] = {}
            for summary in summaries:
                current_value = summary.metrics.get(metric_name)
                if not _is_numeric(current_value) or not _is_numeric(baseline_value):
                    metric_delta[summary.generation_id] = None
                    continue
                metric_delta[summary.generation_id] = float(current_value) - float(
                    baseline_value
                )
            deltas[metric_name] = metric_delta

        return deltas

    def _load_core_models(self) -> None:
        """Load persisted core model records if configured."""

        if self._core_models_path is None or not self._core_models_path.exists():
            return

        payload = json.loads(self._core_models_path.read_text(encoding="utf-8"))
        models = list(payload.get("models", []))
        with self._lock:
            self._core_models = {
                str(item["model_id"]): CoreModelRecord.from_dict(dict(item))
                for item in models
            }

    def _persist_core_models(self) -> None:
        """Persist core model records to disk when configured."""

        if self._core_models_path is None:
            return

        self._core_models_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "saved_at": datetime.now(tz=UTC).isoformat(),
            "models": [record.to_dict() for record in self._core_models.values()],
        }
        self._core_models_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _matches_type(value: object, expected: type[int | float | str | bool]) -> bool:
    """Check primitive type compatibility for hyperparameter validation."""

    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected is float:
        return isinstance(value, _NUMERIC_TYPES) and not isinstance(value, bool)
    if expected is str:
        return isinstance(value, str)
    if expected is bool:
        return isinstance(value, bool)
    return False


def _is_numeric(value: object) -> bool:
    """Return True when value is numeric and not a boolean."""

    return isinstance(value, _NUMERIC_TYPES) and not isinstance(value, bool)


def _parse_datetime(value: str) -> datetime:
    """Parse ISO datetime and normalize to UTC-aware values."""

    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _paginate(
    items: list[ResponseItemT],
    page: int,
    page_size: int,
) -> PaginatedResponse[ResponseItemT]:
    """Return a consistent paginated response object."""

    if page < 1:
        raise ModelValidationError("page must be greater than or equal to 1")
    if page_size < 1:
        raise ModelValidationError("page_size must be greater than or equal to 1")

    total = len(items)
    pages = ceil(total / page_size) if total > 0 else 0

    start = (page - 1) * page_size
    end = start + page_size

    if page > 1 and start >= total and total > 0:
        raise ModelValidationError(
            f"Requested page {page} exceeds available pages ({pages})"
        )

    return PaginatedResponse[ResponseItemT](
        items=items[start:end],
        total=total,
        page=page,
        page_size=page_size,
        pages=pages,
    )


def create_default_model_service(project_root: Path | None = None) -> ModelService:
    """Build a ModelService instance with filesystem-backed defaults."""

    root = project_root or Path(__file__).resolve().parents[4]
    data_dir = root / "data" / "api"
    strategies_dir = root / "scripts" / "custom_functions" / "strategies"

    supporting_registry_path = data_dir / "supporting_models_registry.json"
    core_models_path = data_dir / "core_models_registry.json"
    generations_storage_path = data_dir / "generation_history"

    strategy_dirs = [str(strategies_dir)] if strategies_dir.exists() else []

    supporting_registry = SupportingModelRegistry(
        config_path=str(supporting_registry_path)
    )
    strategy_registry = CustomStrategyRegistry(
        strategy_dirs=strategy_dirs, auto_scan=True
    )
    generation_tracker = GenerationTracker(
        JsonFileStorage(str(generations_storage_path))
    )

    return ModelService(
        supporting_registry=supporting_registry,
        strategy_registry=strategy_registry,
        generation_tracker=generation_tracker,
        core_models_path=core_models_path,
    )


def get_model_service(request: Request) -> ModelService:
    """FastAPI dependency resolver for the shared ModelService instance."""

    service = getattr(request.app.state, "model_service", None)
    if service is None:
        service = create_default_model_service()
        request.app.state.model_service = service
    return service


__all__ = [
    "ModelService",
    "ModelServiceError",
    "ModelValidationError",
    "ModelNotFoundError",
    "InvalidModelStateError",
    "StrategyNotFoundError",
    "GenerationNotFoundError",
    "create_default_model_service",
    "get_model_service",
]
