"""EMA crossover momentum strategy implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from algotrading.src.metrics.base_metrics import BaseMetrics


class EmaMomentum(BaseMetrics):
    """Trend-following strategy on fast/slow EMA separation.

    The close window arrives MinMax-scaled per observation window, so the
    signal is the EMA separation normalised by the window's own dispersion —
    an affine-invariant momentum score that survives the per-step
    re-scaling. Entries require the score to clear ``ENTRY_SCORE`` so the
    strategy stays out of chop where round-trip costs dominate; exits
    combine momentum fade (with hysteresis via ``EXIT_SCORE``) with hard
    stop-loss / take-profit checks on ``trade_change``, which is expressed
    in real cost-adjusted percent.
    """

    FAST_SPAN = 8
    """Span of the fast EMA in bars."""

    SLOW_SPAN = 21
    """Span of the slow EMA in bars; also the minimum usable window."""

    ENTRY_SCORE = 0.25
    """Normalised EMA separation required to open a position."""

    EXIT_SCORE = 0.05
    """Normalised EMA separation below which momentum is considered faded."""

    STOP_LOSS_PCT = -0.35
    """Close the open trade when its net percent change falls to this level."""

    TAKE_PROFIT_PCT = 0.90
    """Close the open trade when its net percent change reaches this level."""

    MIN_STD = 1e-6
    """Dispersion floor below which the window is treated as signal-free."""

    HOLD = 0
    BUY = 1
    SELL = 2

    def predict(self, state: dict[str, np.ndarray], _states: dict) -> tuple:
        """Return the trend-following action for the current state window."""

        closes = np.asarray(state["close"], dtype=np.float64)
        position = int(state["current_position"][-1])
        trade_change = float(state["trade_change"][-1])

        score = self._momentum_score(closes)
        action = self._select_action(position, trade_change, score)

        self.logger.info(f"ema momentum score: {score}, action: {action}")

        return np.array([action], dtype=np.int64), {"score": score}

    def _momentum_score(self, closes: np.ndarray) -> float:
        """Compute the dispersion-normalised fast/slow EMA separation."""

        if closes.size < self.SLOW_SPAN or not np.isfinite(closes).all():
            return 0.0

        series = pd.Series(closes)
        ema_fast = float(series.ewm(span=self.FAST_SPAN, adjust=False).mean().iloc[-1])
        ema_slow = float(series.ewm(span=self.SLOW_SPAN, adjust=False).mean().iloc[-1])
        dispersion = float(series.std())

        if not np.isfinite(dispersion) or dispersion < self.MIN_STD:
            return 0.0

        return (ema_fast - ema_slow) / dispersion

    def _select_action(
        self, position: int, trade_change: float, score: float
    ) -> int:
        """Map position, open-trade profit, and momentum score to an action."""

        if position == 0:
            if score >= self.ENTRY_SCORE:
                return self.BUY
            if score <= -self.ENTRY_SCORE:
                return self.SELL
            return self.HOLD

        stop_or_take = (
            trade_change <= self.STOP_LOSS_PCT or trade_change >= self.TAKE_PROFIT_PCT
        )

        if position == 1:
            if stop_or_take or score <= self.EXIT_SCORE:
                return self.SELL
            return self.HOLD

        if position == 2:
            if stop_or_take or score >= -self.EXIT_SCORE:
                return self.BUY
            return self.HOLD

        return self.HOLD
