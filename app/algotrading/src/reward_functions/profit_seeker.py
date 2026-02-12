"""Profit-based reward calculation implementation."""

from __future__ import annotations

import math

from algotrading.src.constants import RewardConstants as RC
from algotrading.src.metrics.base_metrics import BaseMetrics
from algotrading.src.reward_functions.reward_wrapper import reward_wrapper_function


class ProfitSeeker(BaseMetrics):
    """Reward calculator using profit and position signals."""

    @reward_wrapper_function
    def calculate_reward(self, state: dict) -> float:
        """Calculate reward based on trade and running profit metrics."""

        base_reward = RC.BASE_REWARD

        trade_change = state["trade_change"][-1]
        position_change = state["trade_change"][-2]
        running_change = state["running_profit"][-1]

        if trade_change > 0:
            trade_profit_reward = trade_change * RC.TRADE_PROFIT_MULTIPLIER
        elif trade_change < 0:
            trade_profit_reward = trade_change
        else:
            trade_profit_reward = 0

        if position_change > 0:
            position_reward = position_change * RC.POSITION_CHANGE_MULTIPLIER
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
            "hold_nothing": RC.ACTION_REWARD_HOLD_NOTHING,
            "hold_long_position": RC.ACTION_REWARD_HOLD_POSITION + trade_profit_reward,
            "hold_short_position": RC.ACTION_REWARD_HOLD_POSITION + trade_profit_reward,
            "buy_long": RC.ACTION_REWARD_BUY_LONG,
            "sell_short": RC.ACTION_REWARD_SELL_SHORT,
            "sell_position": RC.ACTION_REWARD_SELL_POSITION + position_reward,
            "buyback_short": RC.ACTION_REWARD_BUYBACK + position_reward,
            "false_buy": RC.ACTION_PENALTY_FALSE_BUY,
            "false_sell": RC.ACTION_PENALTY_FALSE_SELL,
        }

        action_reward = action_reward_dict[self.action_type]

        reward = base_reward + action_reward + running_profit_reward

        self.logger.info(f"step reward: {reward}")

        return reward
