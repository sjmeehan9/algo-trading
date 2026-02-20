"""Trading environment for RL training."""

from __future__ import annotations

from typing import Any

from algotrading.src.envs.base_trading_env import BaseTradingEnv


class TradingEnv(BaseTradingEnv):
    """Gymnasium environment with reward calculation."""

    def _get_reward(self, action: int, state: dict[str, Any]) -> float:
        """Calculate reward using the configured custom logic."""

        return float(self.state_builder.custom_logic.calculate_reward(state))
