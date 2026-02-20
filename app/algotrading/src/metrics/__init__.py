"""Metrics module exports.

Provides ``BaseMetrics`` eagerly and lazy aliases ``RewardCalculator``
(ProfitSeeker) and ``StrategyMetrics`` (ProfitMetrics) to avoid circular
imports — both subclass files import from this package.
"""

from algotrading.src.metrics.base_metrics import BaseMetrics

__all__ = ["BaseMetrics", "RewardCalculator", "StrategyMetrics"]


def __getattr__(name: str) -> type:
    """Lazily resolve subclass aliases to break circular imports."""

    if name == "RewardCalculator":
        from algotrading.src.reward_functions.profit_seeker import ProfitSeeker

        return ProfitSeeker

    if name == "StrategyMetrics":
        from algotrading.src.strategies.profit_metrics import ProfitMetrics

        return ProfitMetrics

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
