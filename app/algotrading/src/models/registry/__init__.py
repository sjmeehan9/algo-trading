"""Public API for supporting model registry package."""

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
from algotrading.src.models.registry.strategy_decorator import (
    StrategyProtocol,
    trading_strategy,
)
from algotrading.src.models.registry.strategy_loader import StrategyLoader
from algotrading.src.models.registry.strategy_registry import (
    CustomStrategyRegistry,
    StrategyEntry,
)
from algotrading.src.models.registry.supporting_model_registry import (
    SupportingModelRegistry,
)

__all__ = [
    "SupportingModelRegistry",
    "ModelEntry",
    "ModelEntryConfig",
    "ModelState",
    "StrategyProtocol",
    "trading_strategy",
    "StrategyLoader",
    "StrategyEntry",
    "CustomStrategyRegistry",
    "RegistryError",
    "ModelNotFoundError",
    "ModelAlreadyExistsError",
    "InvalidDependencyError",
    "ModelStateError",
]
