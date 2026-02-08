"""Trading environment for RL training."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from algotrading.src.envs.base_trading_env import BaseTradingEnv
from algotrading.src.trading.tools import TradingTools
from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete


class TradingEnv(BaseTradingEnv):
    """Gymnasium environment with reward calculation."""

    def _get_reward(self, action: int, state: dict[str, Any]) -> float:
        """Calculate reward using the configured custom logic."""

        return float(self.state_builder.custom_logic.calculate_reward(state))
