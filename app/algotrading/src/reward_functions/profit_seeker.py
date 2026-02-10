"""Profit-based reward calculation implementation."""

from __future__ import annotations

import math

from algotrading.src.metrics.base_metrics import BaseMetrics
from algotrading.src.reward_functions.reward_wrapper import reward_wrapper_function


class ProfitSeeker(BaseMetrics):
    """Reward calculator using profit and position signals."""

    @reward_wrapper_function
    def calculate_reward(self, state: dict) -> float:
        """Calculate reward based on trade and running profit metrics."""

        base_reward = 0

        trade_change = state["trade_change"][-1]
        position_change = state["trade_change"][-2]
        running_change = state["running_profit"][-1]

        if trade_change > 0:
            trade_profit_reward = trade_change * 5
        elif trade_change < 0:
            trade_profit_reward = trade_change
        else:
            trade_profit_reward = 0

        if position_change > 0:
            position_reward = position_change * 500
        elif position_change < 0:
            position_reward = 0
        else:
            position_reward = 0

        if running_change > 0:
            running_profit = math.ceil(running_change)
            running_profit_reward = running_profit**3
        else:
            running_profit_reward = 0

        action_reward_dict = {
            "hold_nothing": 0.5,
            "hold_long_position": 1.5 + trade_profit_reward,
            "hold_short_position": 1.5 + trade_profit_reward,
            "buy_long": 1,
            "sell_short": 1,
            "sell_position": 1 + position_reward,
            "buyback_short": 1 + position_reward,
            "false_buy": -10,
            "false_sell": -10,
        }

        action_reward = action_reward_dict[self.action_type]

        reward = base_reward + action_reward + running_profit_reward

        self.logger.info(f"step reward: {reward}")

        return reward
