"""Trading environment for strategy evaluation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from algotrading.src.envs.base_trading_env import BaseTradingEnv
from algotrading.src.trading.tools import TradingTools
from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete


class StrategyEnv(BaseTradingEnv):
    """Gymnasium environment that returns a dummy reward."""

    def _get_reward(self, action: int, state: dict[str, Any]) -> float:
        """Return a fixed reward for strategy evaluation."""

        return float(self.DUMMY_REWARD)
