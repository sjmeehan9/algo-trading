"""Custom strategy registry for signal-producing rule-based strategies."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Protocol

from algotrading.src.models.registry.strategy_loader import StrategyLoader
from algotrading.src.models.signals import SignalType

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StrategyEntry:
    """Runtime entry for a discovered strategy."""

    strategy_id: str
    name: str
    strategy_class: type[object]
    signal_type: SignalType
    description: str
    version: str
    filepath: Path
    instance: object | None
    last_modified: datetime


class _SignalTypeProtocol(Protocol):
    """Protocol for signal type values exposed on model entries."""

    value: str


class _ModelConfigProtocol(Protocol):
    """Protocol for supporting model config entries used in unified views."""

    model_id: str
    model_type: str
    signal_type: _SignalTypeProtocol
    description: str | None


class _ModelStateProtocol(Protocol):
    """Protocol for model state values exposed on registry entries."""

    value: str


class _ModelEntryProtocol(Protocol):
    """Protocol for minimal model entry shape used by unified view exports."""

    config: _ModelConfigProtocol
    state: _ModelStateProtocol


class _ModelRegistryProtocol(Protocol):
    """Structural type for model registries used in unified strategy views."""

    def get_all(self) -> list[_ModelEntryProtocol]:
        """Return model entries from the supporting model registry."""
        ...


class CustomStrategyRegistry:
    """Registry for discovering and managing decorated custom strategies."""

    def __init__(self, strategy_dirs: list[str], auto_scan: bool = True) -> None:
        """Initialize strategy registry.

        Args:
            strategy_dirs: Filesystem directories scanned for decorated strategies.
            auto_scan: Whether to perform an initial scan during construction.
        """

        self._entries: dict[str, StrategyEntry] = {}
        self._lock = RLock()
        self._loader = StrategyLoader(strategy_dirs)

        if auto_scan:
            self.scan_and_register()

    def scan_and_register(self) -> int:
        """Scan configured directories and register discovered strategies."""

        registered = 0
        for strategy_class, _, filepath in self._loader.load_all():
            try:
                strategy_id = f"{strategy_class.__module__}.{strategy_class.__name__}"
                existed_before = strategy_id in self._entries
                self.register_strategy(strategy_class, filepath=filepath)
                if not existed_before:
                    registered += 1
            except ValueError as exc:
                logger.error(
                    "Skipping strategy %s due to registration error: %s",
                    strategy_class,
                    exc,
                )
        return registered

    def register_strategy(
        self,
        strategy_class: type[object],
        filepath: Path | None = None,
    ) -> str:
        """Register one strategy class and return its ID."""

        metadata_raw = getattr(strategy_class, "_strategy_metadata", None)
        if not isinstance(metadata_raw, dict) or not metadata_raw.get(
            "is_trading_strategy"
        ):
            raise ValueError(
                f"Strategy '{strategy_class.__name__}' is missing @trading_strategy metadata"
            )

        signal_type = metadata_raw.get("signal_type", SignalType.CUSTOM)
        if not isinstance(signal_type, SignalType):
            raise ValueError(
                f"Strategy '{strategy_class.__name__}' has invalid signal_type metadata"
            )

        strategy_id = f"{strategy_class.__module__}.{strategy_class.__name__}"
        resolved_path = (filepath or Path(inspect.getfile(strategy_class))).resolve()
        modified_at = datetime.fromtimestamp(resolved_path.stat().st_mtime, tz=UTC)

        entry = StrategyEntry(
            strategy_id=strategy_id,
            name=str(metadata_raw.get("name", strategy_class.__name__)),
            strategy_class=strategy_class,
            signal_type=signal_type,
            description=str(metadata_raw.get("description", "")).strip(),
            version=str(metadata_raw.get("version", "1.0.0")),
            filepath=resolved_path,
            instance=None,
            last_modified=modified_at,
        )

        with self._lock:
            self._entries[strategy_id] = entry

        return strategy_id

    def unregister(self, strategy_id: str) -> bool:
        """Remove a strategy from the registry."""

        with self._lock:
            if strategy_id not in self._entries:
                return False
            self._entries.pop(strategy_id)
            return True

    def get(self, strategy_id: str) -> StrategyEntry | None:
        """Get a strategy entry by ID."""

        return self._entries.get(strategy_id)

    def get_all(self) -> list[StrategyEntry]:
        """Return all registered strategy entries."""

        return list(self._entries.values())

    def find_by_signal_type(self, signal_type: SignalType) -> list[StrategyEntry]:
        """Filter strategies by emitted signal type."""

        return [
            entry
            for entry in self._entries.values()
            if entry.signal_type == signal_type
        ]

    def get_instance(self, strategy_id: str) -> object:
        """Get or lazily create a strategy instance."""

        with self._lock:
            entry = self._entries.get(strategy_id)
            if entry is None:
                raise KeyError(f"Strategy '{strategy_id}' is not registered")
            if entry.instance is None:
                entry.instance = entry.strategy_class()
            return entry.instance

    def reset_instance(self, strategy_id: str) -> None:
        """Reset strategy instance state if reset method is implemented."""

        with self._lock:
            entry = self._entries.get(strategy_id)
            if entry is None:
                raise KeyError(f"Strategy '{strategy_id}' is not registered")
            if entry.instance is None:
                return

            reset_method = getattr(entry.instance, "reset", None)
            if callable(reset_method):
                reset_method()

    def check_for_changes(self) -> list[str]:
        """Detect file changes and hot-reload modified strategies."""

        changed: list[str] = []
        for strategy_id, entry in list(self._entries.items()):
            try:
                modified_at = datetime.fromtimestamp(
                    entry.filepath.stat().st_mtime, tz=UTC
                )
            except FileNotFoundError:
                continue

            if modified_at > entry.last_modified:
                self.reload_strategy(strategy_id)
                changed.append(strategy_id)

        return changed

    def reload_strategy(self, strategy_id: str) -> None:
        """Reload a strategy from source and replace the registry entry."""

        with self._lock:
            entry = self._entries.get(strategy_id)
            if entry is None:
                raise KeyError(f"Strategy '{strategy_id}' is not registered")

            module = self._loader.load_module(entry.filepath)
            candidates = self._loader.find_strategies(module)
            class_name = entry.strategy_class.__name__
            replacement = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.__name__ == class_name
                ),
                None,
            )
            if replacement is None:
                raise ValueError(
                    f"Strategy class '{class_name}' not found during reload for {strategy_id}"
                )

            metadata_raw = getattr(replacement, "_strategy_metadata", {})
            signal_type = metadata_raw.get("signal_type", SignalType.CUSTOM)
            if not isinstance(signal_type, SignalType):
                raise ValueError(
                    f"Strategy '{replacement.__name__}' has invalid signal_type metadata"
                )

            entry.strategy_class = replacement
            entry.name = str(metadata_raw.get("name", replacement.__name__))
            entry.description = str(metadata_raw.get("description", "")).strip()
            entry.version = str(metadata_raw.get("version", "1.0.0"))
            entry.signal_type = signal_type
            entry.instance = None
            entry.last_modified = datetime.fromtimestamp(
                entry.filepath.stat().st_mtime,
                tz=UTC,
            )

    def export_model_entries(self) -> list[dict[str, object]]:
        """Export strategies in a model-entry-compatible dictionary shape."""

        entries: list[dict[str, object]] = []
        for entry in self._entries.values():
            entries.append(
                {
                    "model_id": entry.strategy_id,
                    "model_type": "strategy",
                    "signal_type": entry.signal_type.value,
                    "description": entry.description,
                    "version": entry.version,
                    "state": "ready",
                    "filepath": str(entry.filepath),
                    "name": entry.name,
                }
            )
        return entries

    def get_unified_view(
        self,
        model_registry: _ModelRegistryProtocol | None = None,
    ) -> list[dict[str, object]]:
        """Return one unified list containing model and strategy providers."""

        combined: list[dict[str, object]] = []

        if model_registry is not None:
            for model_entry in model_registry.get_all():
                combined.append(
                    {
                        "model_id": model_entry.config.model_id,
                        "model_type": model_entry.config.model_type,
                        "signal_type": model_entry.config.signal_type.value,
                        "description": model_entry.config.description,
                        "state": model_entry.state.value,
                    }
                )

        combined.extend(self.export_model_entries())
        return combined


__all__ = ["CustomStrategyRegistry", "StrategyEntry"]
