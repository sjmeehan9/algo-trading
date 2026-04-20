"""Strategy file discovery, dynamic loading, and validation utilities."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import logging
from pathlib import Path
from types import ModuleType

from algotrading.src.models.signals import ModelSignal

logger = logging.getLogger(__name__)


class StrategyLoader:
    """Discover and load decorated strategy classes from source files."""

    def __init__(self, strategy_dirs: list[str]) -> None:
        """Initialize loader with one or more directories to scan."""

        self._strategy_dirs = [Path(value).resolve() for value in strategy_dirs]

    def scan_directories(self) -> list[Path]:
        """Find Python source files under configured strategy directories."""

        files: set[Path] = set()
        for directory in self._strategy_dirs:
            if not directory.exists() or not directory.is_dir():
                continue

            for path in directory.rglob("*.py"):
                if path.name.startswith("__"):
                    continue
                files.add(path.resolve())

        return sorted(files)

    def load_module(self, filepath: Path) -> ModuleType:
        """Dynamically import a module from a file path."""

        resolved = filepath.resolve()
        module_name = self._build_module_name(resolved)
        spec = importlib.util.spec_from_file_location(module_name, resolved)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to create module spec for '{resolved}'")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def find_strategies(self, module: ModuleType) -> list[type[object]]:
        """Find classes marked by ``@trading_strategy`` in a loaded module."""

        strategies: list[type[object]] = []
        for _, member in inspect.getmembers(module, inspect.isclass):
            metadata = getattr(member, "_strategy_metadata", None)
            if isinstance(metadata, dict) and metadata.get("is_trading_strategy"):
                strategies.append(member)
        return strategies

    def validate_strategy(self, strategy_class: type[object]) -> tuple[bool, list[str]]:
        """Validate strategy contract conformance."""

        errors: list[str] = []

        metadata = getattr(strategy_class, "_strategy_metadata", None)
        if not isinstance(metadata, dict) or not metadata.get("is_trading_strategy"):
            errors.append("Missing @trading_strategy decorator")

        predict = getattr(strategy_class, "predict", None)
        if predict is None or not callable(predict):
            errors.append("Missing predict() method")
            return False, errors

        signature = inspect.signature(predict)
        parameter_names = list(signature.parameters.keys())
        if len(parameter_names) < 3:
            errors.append(
                "predict() must accept (self, state, market_data) as minimum parameters"
            )

        return_annotation = signature.return_annotation
        if return_annotation not in (inspect.Signature.empty, ModelSignal):
            errors.append(
                "predict() return annotation must be ModelSignal when provided"
            )

        return len(errors) == 0, errors

    def load_all(self) -> list[tuple[type[object], dict[str, object], Path]]:
        """Load and validate all discoverable strategies.

        Returns:
            Tuples containing strategy class, metadata, and source file path.
        """

        loaded: list[tuple[type[object], dict[str, object], Path]] = []
        for file_path in self.scan_directories():
            try:
                module = self.load_module(file_path)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to import strategy module %s: %s", file_path, exc)
                continue

            for strategy_class in self.find_strategies(module):
                is_valid, errors = self.validate_strategy(strategy_class)
                if not is_valid:
                    logger.error(
                        "Invalid strategy class '%s' in %s: %s",
                        strategy_class.__name__,
                        file_path,
                        "; ".join(errors),
                    )
                    continue

                metadata_raw = getattr(strategy_class, "_strategy_metadata", {})
                metadata = {
                    "name": str(metadata_raw.get("name", strategy_class.__name__)),
                    "signal_type": metadata_raw.get("signal_type"),
                    "description": str(metadata_raw.get("description", "")).strip(),
                    "version": str(metadata_raw.get("version", "1.0.0")),
                }
                loaded.append((strategy_class, metadata, file_path))

        return loaded

    def _build_module_name(self, filepath: Path) -> str:
        digest = hashlib.sha1(str(filepath).encode("utf-8"), usedforsecurity=False)
        return f"algotrading_user_strategy_{digest.hexdigest()}"


__all__ = ["StrategyLoader"]
