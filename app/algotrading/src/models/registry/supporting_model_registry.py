"""Supporting model registry for ML and RL signal providers."""

from __future__ import annotations

import importlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Callable

from algotrading.src.data_pipeline import DataType
from algotrading.src.models.registry.exceptions import (
    InvalidDependencyError,
    ModelAlreadyExistsError,
    ModelNotFoundError,
    ModelStateError,
    RegistryError,
)
from algotrading.src.models.registry.model_entry import (
    ModelEntry,
    ModelEntryConfig,
    ModelState,
)
from algotrading.src.models.signals import SignalType

logger = logging.getLogger(__name__)

_ALLOWED_SUPPORTING_INPUT_TYPES = {
    DataType.MARKET_BAR,
    DataType.NEWS_TEXT,
    DataType.INDICATOR,
}

_STATE_TRANSITIONS: dict[ModelState, set[ModelState]] = {
    ModelState.REGISTERED: {
        ModelState.REGISTERED,
        ModelState.LOADING,
        ModelState.UNLOADED,
        ModelState.ERROR,
    },
    ModelState.LOADING: {ModelState.LOADED, ModelState.ERROR},
    ModelState.LOADED: {
        ModelState.LOADED,
        ModelState.READY,
        ModelState.UNLOADED,
        ModelState.ERROR,
    },
    ModelState.READY: {
        ModelState.READY,
        ModelState.LOADED,
        ModelState.UNLOADED,
        ModelState.ERROR,
    },
    ModelState.ERROR: {
        ModelState.ERROR,
        ModelState.REGISTERED,
        ModelState.LOADING,
        ModelState.UNLOADED,
    },
    ModelState.UNLOADED: {
        ModelState.UNLOADED,
        ModelState.LOADING,
        ModelState.REGISTERED,
    },
}


class SupportingModelRegistry:
    """Thread-safe registry for supporting ML/RL models.

    The registry stores model configurations, enforces dependency constraints,
    tracks runtime lifecycle state, and supports persistence to JSON.
    """

    def __init__(self, config_path: str | None = None) -> None:
        """Initialize an empty registry.

        Args:
            config_path: Optional path used for auto-saving registry configuration.
        """

        self._entries: dict[str, ModelEntry] = {}
        self._lock = RLock()
        self._config_path = config_path
        self._state_callbacks: list[Callable[[str, ModelState], None]] = []

        if self._config_path:
            path = Path(self._config_path)
            if path.exists():
                self.load_config(str(path))

    def register(self, config: ModelEntryConfig) -> str:
        """Register a supporting model.

        Args:
            config: Model registration configuration.

        Returns:
            The registered model ID.

        Raises:
            ModelAlreadyExistsError: If model ID is already present.
            InvalidDependencyError: If input dependencies violate architecture rules.
        """

        with self._lock:
            self._validate_no_supporting_dependencies(config)
            if config.model_id in self._entries:
                raise ModelAlreadyExistsError(
                    f"Model '{config.model_id}' is already registered"
                )

            self._entries[config.model_id] = ModelEntry(config=config)
            self._auto_save()
            self._emit_state_change(config.model_id, ModelState.REGISTERED)
            return config.model_id

    def register_from_config(self, config_dict: dict[str, object]) -> str:
        """Register a model from a dictionary payload."""

        return self.register(ModelEntryConfig.from_dict(config_dict))

    def unregister(self, model_id: str) -> bool:
        """Unregister a model from the registry."""

        with self._lock:
            existed = model_id in self._entries
            if not existed:
                return False

            self._entries.pop(model_id)
            self._auto_save()
            return True

    def update_config(self, model_id: str, updates: dict[str, object]) -> None:
        """Update mutable model configuration fields."""

        with self._lock:
            entry = self._entries.get(model_id)
            if entry is None:
                raise ModelNotFoundError(f"Model '{model_id}' not found")

            merged = entry.config.to_dict()
            merged.update(updates)
            merged["model_id"] = entry.config.model_id

            updated = ModelEntryConfig.from_dict(merged)
            self._validate_no_supporting_dependencies(updated)
            entry.config = updated
            self._auto_save()

    def get(self, model_id: str) -> ModelEntry | None:
        """Get a model entry by ID."""

        return self._entries.get(model_id)

    def get_all(self) -> list[ModelEntry]:
        """Return all registered model entries."""

        return list(self._entries.values())

    def find_by_type(self, model_type: str) -> list[ModelEntry]:
        """Find registered models by model type (``ml`` or ``rl``)."""

        normalized = model_type.strip().lower()
        return [
            entry
            for entry in self._entries.values()
            if entry.config.model_type == normalized
        ]

    def find_by_signal_type(self, signal_type: SignalType) -> list[ModelEntry]:
        """Find registered models by output signal type."""

        return [
            entry
            for entry in self._entries.values()
            if entry.config.signal_type == signal_type
        ]

    def find_ready(self) -> list[ModelEntry]:
        """Return entries currently in ``READY`` state."""

        return [
            entry for entry in self._entries.values() if entry.state == ModelState.READY
        ]

    def load_model(self, model_id: str) -> None:
        """Instantiate and attach the trainer class for a model entry."""

        with self._lock:
            entry = self._require_entry(model_id)
            self.set_state(model_id, ModelState.LOADING)

            try:
                trainer_cls = self._resolve_trainer_class(entry.config.trainer_class)
                trainer = trainer_cls()
                entry.trainer = trainer
                entry.loaded_at = datetime.now(tz=UTC)
                entry.error_message = None
                self.set_state(model_id, ModelState.LOADED)
            except Exception as exc:
                self.set_state(model_id, ModelState.ERROR, error=str(exc))
                raise RegistryError(
                    f"Failed to load trainer for model '{model_id}': {exc}"
                ) from exc

    def unload_model(self, model_id: str) -> None:
        """Unload model trainer instance and mark entry as unloaded."""

        with self._lock:
            entry = self._require_entry(model_id)
            entry.trainer = None
            entry.loaded_at = None
            self.set_state(model_id, ModelState.UNLOADED)

    def get_state(self, model_id: str) -> ModelState:
        """Return current lifecycle state for the given model ID."""

        return self._require_entry(model_id).state

    def set_state(
        self, model_id: str, state: ModelState, error: str | None = None
    ) -> None:
        """Set model state with transition validation and callback emission."""

        with self._lock:
            entry = self._require_entry(model_id)
            self._validate_state_transition(entry.state, state, model_id)
            entry.state = state
            entry.error_message = error if state == ModelState.ERROR else None
            self._auto_save()

        self._emit_state_change(model_id, state)

    def save_config(self, filepath: str | None = None) -> None:
        """Persist registry entries to JSON file."""

        resolved_path = filepath or self._config_path
        if resolved_path is None:
            raise RegistryError("No filepath provided and no registry config_path set")
        output_path = Path(resolved_path)

        payload = {
            "saved_at": datetime.now(tz=UTC).isoformat(),
            "models": [entry.to_dict() for entry in self._entries.values()],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def load_config(self, filepath: str) -> None:
        """Load registry entries from a JSON file."""

        path = Path(filepath)
        if not path.exists():
            raise RegistryError(f"Registry config file does not exist: {filepath}")

        raw = json.loads(path.read_text(encoding="utf-8"))
        models = list(raw.get("models", []))

        with self._lock:
            loaded: dict[str, ModelEntry] = {}
            for item in models:
                entry = ModelEntry.from_dict(dict(item))
                self._validate_no_supporting_dependencies(entry.config)
                loaded[entry.config.model_id] = entry
            self._entries = loaded

    def on_state_change(self, callback: Callable[[str, ModelState], None]) -> None:
        """Register callback fired when a model state changes."""

        with self._lock:
            self._state_callbacks.append(callback)

    def _validate_no_supporting_dependencies(self, config: ModelEntryConfig) -> None:
        """Ensure supporting models do not consume other supporting model signals."""

        if DataType.SIGNAL in config.input_data_types:
            raise InvalidDependencyError(
                "Supporting models cannot depend on DataType.SIGNAL inputs"
            )

        disallowed = set(config.input_data_types) - _ALLOWED_SUPPORTING_INPUT_TYPES
        if disallowed:
            value = ", ".join(sorted(item.value for item in disallowed))
            raise InvalidDependencyError(
                "Unsupported supporting model input data types: " f"{value}"
            )

    def _resolve_trainer_class(self, trainer_class_path: str) -> type[object]:
        """Resolve a fully-qualified trainer class path."""

        try:
            module_name, class_name = trainer_class_path.rsplit(".", 1)
        except ValueError as exc:
            raise RegistryError(
                f"Invalid trainer_class path '{trainer_class_path}'"
            ) from exc

        module = importlib.import_module(module_name)
        trainer_class = getattr(module, class_name)
        if not isinstance(trainer_class, type):
            raise RegistryError(
                f"Resolved trainer class '{trainer_class_path}' is not a class"
            )
        return trainer_class

    def _require_entry(self, model_id: str) -> ModelEntry:
        """Return entry or raise ``ModelNotFoundError``."""

        entry = self._entries.get(model_id)
        if entry is None:
            raise ModelNotFoundError(f"Model '{model_id}' not found")
        return entry

    def _validate_state_transition(
        self,
        current: ModelState,
        requested: ModelState,
        model_id: str,
    ) -> None:
        """Validate requested state transition."""

        allowed = _STATE_TRANSITIONS.get(current, set())
        if requested not in allowed:
            raise ModelStateError(
                f"Invalid state transition for '{model_id}': "
                f"{current.value} -> {requested.value}"
            )

    def _auto_save(self) -> None:
        """Persist registry when auto-save path is configured."""

        if self._config_path is None:
            return
        self.save_config(self._config_path)

    def _emit_state_change(self, model_id: str, state: ModelState) -> None:
        """Invoke state callbacks without letting callback failures bubble."""

        callbacks = list(self._state_callbacks)
        for callback in callbacks:
            try:
                callback(model_id, state)
            except Exception as exc:
                logger.exception(
                    "State change callback failed for model '%s': %s",
                    model_id,
                    exc,
                )


__all__ = ["SupportingModelRegistry"]
