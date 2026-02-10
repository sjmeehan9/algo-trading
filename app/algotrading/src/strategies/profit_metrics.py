"""Profit metrics strategy implementation."""

from __future__ import annotations

import numpy as np
from algotrading.src.metrics.base_metrics import BaseMetrics
from algotrading.src.strategies.strategy_wrapper import strategy_wrapper_function


class ProfitMetrics(BaseMetrics):
    """Strategy metrics implementation with a default action."""

    @strategy_wrapper_function
    def predict(self, state: dict[str, np.ndarray], _states: dict) -> tuple:
        """Return the default action and empty state container."""

        action = 0
        _states = {}
        return action, _states
