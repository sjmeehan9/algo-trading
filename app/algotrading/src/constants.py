"""Named constants for reward calculation and trading logic.

This module centralises magic numbers that were previously scattered
across ``profit_seeker.py`` and ``tools.py``.  Each constant includes a
docstring-level explanation of its role so future developers can understand
the reward-shaping and risk-management design.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Reward constants  (used by ProfitSeeker.calculate_reward)
# ---------------------------------------------------------------------------


class RewardConstants:
    """Constants governing the RL reward function.

    The reward function is additive:
        reward = base_reward + action_reward + running_profit_reward

    ``action_reward`` is selected from the per-action-type table below and
    may itself include a scaled ``trade_profit_reward`` or
    ``position_reward`` term.
    """

    # ------------------------------------------------------------------
    # Base & scaling
    # ------------------------------------------------------------------
    BASE_REWARD: float = 0.0
    """Starting reward value before any bonuses or penalties."""

    TRADE_PROFIT_MULTIPLIER: float = 5.0
    """Scaling factor applied to positive per-trade profit."""

    POSITION_CHANGE_MULTIPLIER: float = 500.0
    """Scaling factor for positive position-change rewards."""

    # ------------------------------------------------------------------
    # Per-action-type bonuses / penalties
    # ------------------------------------------------------------------
    ACTION_REWARD_HOLD_NOTHING: float = 0.5
    """Small positive nudge for staying flat (discourages excessive trading)."""

    ACTION_REWARD_HOLD_POSITION: float = 1.5
    """Bonus for holding an open position (long or short)."""

    ACTION_REWARD_BUY_LONG: float = 1.0
    """Reward for opening a long position."""

    ACTION_REWARD_SELL_SHORT: float = 1.0
    """Reward for opening a short position."""

    ACTION_REWARD_SELL_POSITION: float = 1.0
    """Base reward for closing a long position."""

    ACTION_REWARD_BUYBACK: float = 1.0
    """Base reward for closing a short position."""

    ACTION_PENALTY_FALSE_BUY: float = -10.0
    """Heavy penalty for attempting to buy when already long."""

    ACTION_PENALTY_FALSE_SELL: float = -10.0
    """Heavy penalty for attempting to sell when already short."""


# ---------------------------------------------------------------------------
# Trading / risk-management constants  (used by TradingTools)
# ---------------------------------------------------------------------------


class TradingConstants:
    """Constants governing the stop-loss / take-profit mechanism.

    ``TradingTools.stop_take`` maps raw model actions through a position
    look-up so it can automatically close a position when risk thresholds
    are breached.
    """

    CLOSE_POSITION_MAP: dict[int, int] = {1: 2, 2: 1}
    """Maps current position → closing action (long→sell, short→buy-back)."""

    HOLD_ACTION: int = 0
    """Action value that keeps the current position unchanged."""
