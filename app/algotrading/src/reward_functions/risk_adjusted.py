"""Risk-adjusted reward calculation implementation."""

from __future__ import annotations

import numpy as np

from algotrading.src.metrics.base_metrics import BaseMetrics


class RiskAdjusted(BaseMetrics):
    """Reward calculator balancing net profit against downside risk.

    The per-step reward is built from the change in session equity
    (``running_profit``, already net of commission and spread), minus
    penalties for downside deviation and open drawdown, plus light action
    shaping. Summed over an episode the profit term telescopes to the final
    session profit, so the optimal policy maximises cost-adjusted profit
    while the risk terms steer it toward smoother equity curves.

    Unlike ``ProfitSeeker`` this reward is symmetric (losses hurt as much as
    gains help), bounded by ``REWARD_CLIP`` to keep the PPO value function
    stable, and never compounds cumulative profit into explosive bonuses.
    """

    PROFIT_SCALE = 10.0
    """Reward units per 1% of equity gained in a single step."""

    DOWNSIDE_WEIGHT = 5.0
    """Penalty per 1% of downside deviation in recent step returns."""

    DRAWDOWN_WEIGHT = 0.05
    """Per-step penalty per 1% of open drawdown from the recent equity peak."""

    RISK_WINDOW = 30
    """Number of recent equity observations used for the risk terms."""

    ENTRY_FRICTION = 0.05
    """Small charge for opening a trade, discouraging cost-inefficient churn."""

    INVALID_ACTION_PENALTY = 5.0
    """Penalty for false buy/sell actions that cannot be executed."""

    REWARD_CLIP = 20.0
    """Absolute bound applied to the final reward value."""

    def calculate_reward(self, state: dict) -> float:
        """Calculate reward from equity change, downside risk, and drawdown."""

        equity = np.asarray(state["running_profit"], dtype=np.float64)
        window = equity[-self.RISK_WINDOW :]

        if window.size > 1:
            step_returns = np.diff(window)
            equity_delta = step_returns[-1]
            downside = np.minimum(step_returns, 0.0)
            downside_deviation = float(np.sqrt(np.mean(np.square(downside))))
            drawdown = float(max(np.max(window) - window[-1], 0.0))
        else:
            equity_delta = 0.0
            downside_deviation = 0.0
            drawdown = 0.0

        action_reward_dict = {
            "hold_nothing": 0.0,
            "hold_long_position": 0.0,
            "hold_short_position": 0.0,
            "buy_long": -self.ENTRY_FRICTION,
            "sell_short": -self.ENTRY_FRICTION,
            "sell_position": 0.0,
            "buyback_short": 0.0,
            "false_buy": -self.INVALID_ACTION_PENALTY,
            "false_sell": -self.INVALID_ACTION_PENALTY,
        }

        reward = (
            self.PROFIT_SCALE * equity_delta
            - self.DOWNSIDE_WEIGHT * downside_deviation
            - self.DRAWDOWN_WEIGHT * drawdown
            + action_reward_dict[self.action_type]
        )

        if not np.isfinite(reward):
            reward = 0.0

        reward = float(np.clip(reward, -self.REWARD_CLIP, self.REWARD_CLIP))

        self.logger.info(f"step reward: {reward}")

        return reward
