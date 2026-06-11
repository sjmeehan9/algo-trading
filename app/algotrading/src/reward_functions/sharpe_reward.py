"""Rolling Sharpe ratio reward calculation implementation."""

from __future__ import annotations

import numpy as np

from algotrading.src.metrics.base_metrics import BaseMetrics


class SharpeReward(BaseMetrics):
    """Reward calculator favouring stable returns relative to volatility.

    Each step rewards the rolling Sharpe ratio of recent per-step equity
    changes (``running_profit`` deltas, already net of commission and
    spread). A policy earns sustained reward only by producing consistent
    positive returns; volatile or directionless trading drives the ratio —
    and the reward — toward zero. Staying flat yields zero, so the agent
    must find genuinely repeatable edges to score.

    The volatility floor prevents division blow-ups in near-flat regimes and
    ``REWARD_CLIP`` bounds the signal for value-function stability.
    """

    SHARPE_SCALE = 10.0
    """Scaling applied to the rolling Sharpe ratio."""

    RISK_WINDOW = 30
    """Number of recent equity observations used for the rolling ratio."""

    STD_FLOOR = 1e-4
    """Volatility floor (in percent units) guarding the denominator."""

    ENTRY_FRICTION = 0.05
    """Small charge for opening a trade, discouraging cost-inefficient churn."""

    INVALID_ACTION_PENALTY = 5.0
    """Penalty for false buy/sell actions that cannot be executed."""

    REWARD_CLIP = 10.0
    """Absolute bound applied to the final reward value."""

    def calculate_reward(self, state: dict) -> float:
        """Calculate reward from the rolling Sharpe of step equity changes."""

        equity = np.asarray(state["running_profit"], dtype=np.float64)
        window = equity[-self.RISK_WINDOW :]

        if window.size > 1:
            step_returns = np.diff(window)
            mean_return = float(np.mean(step_returns))
            return_std = float(np.std(step_returns))
            rolling_sharpe = mean_return / (return_std + self.STD_FLOOR)
        else:
            rolling_sharpe = 0.0

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
            self.SHARPE_SCALE * rolling_sharpe + action_reward_dict[self.action_type]
        )

        if not np.isfinite(reward):
            reward = 0.0

        reward = float(np.clip(reward, -self.REWARD_CLIP, self.REWARD_CLIP))

        self.logger.info(f"step reward: {reward}")

        return reward
