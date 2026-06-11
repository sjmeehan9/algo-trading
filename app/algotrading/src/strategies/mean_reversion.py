"""Z-score mean reversion strategy implementation."""

from __future__ import annotations

import numpy as np

from algotrading.src.metrics.base_metrics import BaseMetrics


class MeanReversion(BaseMetrics):
    """Contrarian strategy fading statistically stretched prices.

    The entry signal is the z-score of the latest close against the recent
    lookback window. Z-scores are invariant under the per-window MinMax
    scaling applied to state data, so the signal is unaffected by the
    re-scaling each step. Positions fade extremes — long when price is
    stretched ``Z_ENTRY`` deviations below the mean, short when above — and
    exit when price reverts inside ``Z_EXIT`` or when hard stop-loss /
    take-profit levels on ``trade_change`` (real cost-adjusted percent) are
    hit. A dispersion floor keeps the strategy flat in dead, signal-free
    windows where the z-score would be numerically meaningless.
    """

    LOOKBACK = 30
    """Number of recent bars used for the mean and deviation estimates."""

    Z_ENTRY = 2.0
    """Absolute z-score required to open a contrarian position."""

    Z_EXIT = 0.25
    """Absolute z-score at which price is considered reverted to the mean."""

    STOP_LOSS_PCT = -0.40
    """Close the open trade when its net percent change falls to this level."""

    TAKE_PROFIT_PCT = 0.60
    """Close the open trade when its net percent change reaches this level."""

    MIN_STD = 1e-4
    """Dispersion floor below which the window is treated as signal-free."""

    HOLD = 0
    BUY = 1
    SELL = 2

    def predict(self, state: dict[str, np.ndarray], _states: dict) -> tuple:
        """Return the mean-reversion action for the current state window."""

        closes = np.asarray(state["close"], dtype=np.float64)
        position = int(state["current_position"][-1])
        trade_change = float(state["trade_change"][-1])

        z_score = self._z_score(closes)
        action = self._select_action(position, trade_change, z_score)

        self.logger.info(f"mean reversion z-score: {z_score}, action: {action}")

        return np.array([action], dtype=np.int64), {"z_score": z_score}

    def _z_score(self, closes: np.ndarray) -> float:
        """Compute the z-score of the latest close over the lookback window."""

        window = closes[-self.LOOKBACK :]

        if window.size < 3 or not np.isfinite(window).all():
            return 0.0

        deviation = float(np.std(window))
        if deviation < self.MIN_STD:
            return 0.0

        return float((window[-1] - np.mean(window)) / deviation)

    def _select_action(
        self, position: int, trade_change: float, z_score: float
    ) -> int:
        """Map position, open-trade profit, and z-score to an action."""

        if position == 0:
            if z_score <= -self.Z_ENTRY:
                return self.BUY
            if z_score >= self.Z_ENTRY:
                return self.SELL
            return self.HOLD

        stop_or_take = (
            trade_change <= self.STOP_LOSS_PCT or trade_change >= self.TAKE_PROFIT_PCT
        )

        if position == 1:
            if stop_or_take or z_score >= -self.Z_EXIT:
                return self.SELL
            return self.HOLD

        if position == 2:
            if stop_or_take or z_score <= self.Z_EXIT:
                return self.BUY
            return self.HOLD

        return self.HOLD
