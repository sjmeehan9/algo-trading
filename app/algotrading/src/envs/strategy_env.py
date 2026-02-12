"""Trading environment for strategy evaluation."""

from __future__ import annotations

from typing import Any

from algotrading.src.envs.base_trading_env import BaseTradingEnv


class StrategyEnv(BaseTradingEnv):
    """Gymnasium environment that returns a dummy reward."""

    def _get_reward(self, action: int, state: dict[str, Any]) -> float:
        """Return a fixed reward for strategy evaluation."""

        return float(self.DUMMY_REWARD)
