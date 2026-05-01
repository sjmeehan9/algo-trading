"""Performance metric calculations for API backtest results."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from algotrading.api.schemas.backtesting import (
    EquityPoint,
    PerformanceMetrics,
    TradeAction,
    TradeRecord,
)


class MetricsCalculator:
    """Calculate industry-standard performance metrics for backtests."""

    TRADING_DAYS_PER_YEAR = 252
    RISK_FREE_RATE = 0.02

    def calculate_all(
        self,
        trades: list[TradeRecord],
        equity_curve: list[EquityPoint],
        initial_capital: float,
    ) -> PerformanceMetrics:
        """Calculate all backtest metrics from trades and equity snapshots.

        Args:
            trades: Trade records generated during the backtest.
            equity_curve: Portfolio equity snapshots for the backtest period.
            initial_capital: Starting portfolio value.

        Returns:
            Fully populated performance metrics with finite numeric values.
        """

        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")

        ordered_equity = sorted(equity_curve, key=lambda point: point.timestamp)
        ordered_trades = sorted(
            trades, key=lambda trade: (trade.timestamp, trade.trade_id)
        )
        returns = self._calculate_returns(ordered_equity)
        trade_pnls = self._realized_trade_pnls(ordered_trades)

        total_return = self._total_return(ordered_equity, initial_capital)
        total_return_dollars = self._total_return_dollars(
            ordered_equity, initial_capital
        )
        winning = [value for value in trade_pnls if value > 0]
        losing = [value for value in trade_pnls if value < 0]

        return PerformanceMetrics(
            total_return=_finite(total_return),
            total_return_dollars=_finite(total_return_dollars),
            annualized_return=_finite(
                self._annualized_return(ordered_equity, initial_capital)
            ),
            sharpe_ratio=_finite(self._sharpe_ratio(returns)),
            sortino_ratio=_finite(self._sortino_ratio(returns)),
            max_drawdown=_finite(self._max_drawdown(ordered_equity)),
            max_drawdown_duration_days=self._max_drawdown_duration(ordered_equity),
            volatility=_finite(self._volatility(returns)),
            win_rate=_finite(self._win_rate(trade_pnls)),
            profit_factor=_finite(self._profit_factor(winning, losing)),
            total_trades=len(
                [trade for trade in ordered_trades if trade.action != TradeAction.HOLD]
            ),
            winning_trades=len(winning),
            losing_trades=len(losing),
            average_win=_finite(float(np.mean(winning)) if winning else 0.0),
            average_loss=_finite(float(np.mean(losing)) if losing else 0.0),
            largest_win=_finite(max(winning) if winning else 0.0),
            largest_loss=_finite(min(losing) if losing else 0.0),
            average_trade_duration=_finite(
                self._average_trade_duration_hours(ordered_trades)
            ),
            exposure_time=_finite(self._exposure_time(ordered_trades, ordered_equity)),
        )

    def _calculate_returns(self, equity_curve: list[EquityPoint]) -> pd.Series:
        """Return percentage changes between consecutive equity values."""

        if len(equity_curve) < 2:
            return pd.Series(dtype="float64")
        values = pd.Series(
            [point.portfolio_value for point in equity_curve], dtype="float64"
        )
        returns = values.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        return returns.astype("float64")

    def _total_return(
        self, equity_curve: list[EquityPoint], initial_capital: float
    ) -> float:
        """Calculate total percentage return for a backtest."""

        if not equity_curve:
            return 0.0
        return (equity_curve[-1].portfolio_value - initial_capital) / initial_capital

    def _total_return_dollars(
        self, equity_curve: list[EquityPoint], initial_capital: float
    ) -> float:
        """Calculate absolute profit or loss in account currency."""

        if not equity_curve:
            return 0.0
        return equity_curve[-1].portfolio_value - initial_capital

    def _annualized_return(
        self, equity_curve: list[EquityPoint], initial_capital: float
    ) -> float:
        """Calculate annualized return using elapsed calendar days."""

        if len(equity_curve) < 2:
            return 0.0
        start_value = initial_capital
        end_value = equity_curve[-1].portfolio_value
        if start_value <= 0 or end_value <= 0:
            return 0.0
        elapsed_days = max(
            (equity_curve[-1].timestamp - equity_curve[0].timestamp).total_seconds()
            / 86_400,
            1.0,
        )
        return (end_value / start_value) ** (365.0 / elapsed_days) - 1.0

    def _sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio from periodic returns."""

        if returns.empty or returns.std(ddof=1) == 0 or math.isnan(returns.std(ddof=1)):
            return 0.0
        excess_returns = returns - (self.RISK_FREE_RATE / self.TRADING_DAYS_PER_YEAR)
        return float(
            np.sqrt(self.TRADING_DAYS_PER_YEAR)
            * (excess_returns.mean() / returns.std(ddof=1))
        )

    def _sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate annualized Sortino ratio from periodic returns."""

        if returns.empty:
            return 0.0
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std(ddof=1)
        if downside_returns.empty or downside_std == 0 or math.isnan(downside_std):
            return 0.0
        excess_return = returns.mean() - (
            self.RISK_FREE_RATE / self.TRADING_DAYS_PER_YEAR
        )
        return float(
            np.sqrt(self.TRADING_DAYS_PER_YEAR) * (excess_return / downside_std)
        )

    def _max_drawdown(self, equity_curve: list[EquityPoint]) -> float:
        """Calculate maximum drawdown percentage as a positive value."""

        if not equity_curve:
            return 0.0
        values = pd.Series(
            [point.portfolio_value for point in equity_curve], dtype="float64"
        )
        peaks = values.expanding(min_periods=1).max()
        drawdowns = (values - peaks) / peaks.replace(0, np.nan)
        min_drawdown = drawdowns.replace([np.inf, -np.inf], np.nan).min()
        if pd.isna(min_drawdown):
            return 0.0
        return abs(float(min_drawdown))

    def _max_drawdown_duration(self, equity_curve: list[EquityPoint]) -> int:
        """Calculate longest drawdown duration in calendar days."""

        if len(equity_curve) < 2:
            return 0

        max_duration = 0.0
        peak_value = float("-inf")
        drawdown_started_at: datetime | None = None

        for point in equity_curve:
            if point.portfolio_value >= peak_value:
                if drawdown_started_at is not None:
                    duration = (point.timestamp - drawdown_started_at).total_seconds()
                    max_duration = max(max_duration, duration)
                peak_value = point.portfolio_value
                drawdown_started_at = None
                continue
            if drawdown_started_at is None:
                drawdown_started_at = point.timestamp

        if drawdown_started_at is not None:
            duration = (
                equity_curve[-1].timestamp - drawdown_started_at
            ).total_seconds()
            max_duration = max(max_duration, duration)

        return int(math.ceil(max_duration / 86_400))

    def _volatility(self, returns: pd.Series) -> float:
        """Calculate annualized return volatility."""

        if returns.empty or returns.std(ddof=1) == 0 or math.isnan(returns.std(ddof=1)):
            return 0.0
        return float(returns.std(ddof=1) * np.sqrt(self.TRADING_DAYS_PER_YEAR))

    def _realized_trade_pnls(self, trades: list[TradeRecord]) -> list[float]:
        """Estimate realized P&L for completed long round trips."""

        positions: dict[str, float] = defaultdict(float)
        average_costs: dict[str, float] = defaultdict(float)
        pnls: list[float] = []

        for trade in trades:
            if trade.action == TradeAction.BUY and trade.quantity > 0:
                current_qty = positions[trade.symbol]
                current_cost = average_costs[trade.symbol]
                total_qty = current_qty + trade.quantity
                if total_qty <= 0:
                    positions[trade.symbol] = 0.0
                    average_costs[trade.symbol] = 0.0
                    continue
                average_costs[trade.symbol] = (
                    (current_cost * current_qty)
                    + (trade.price * trade.quantity)
                    + trade.cost
                ) / total_qty
                positions[trade.symbol] = total_qty
                continue

            if trade.action == TradeAction.SELL and trade.quantity > 0:
                open_qty = positions[trade.symbol]
                if open_qty <= 0:
                    continue
                closed_qty = min(open_qty, trade.quantity)
                pnl = (
                    trade.price - average_costs[trade.symbol]
                ) * closed_qty - trade.cost
                pnls.append(float(pnl))
                remaining = open_qty - closed_qty
                positions[trade.symbol] = remaining
                if remaining <= 0:
                    average_costs[trade.symbol] = 0.0

        if pnls:
            return pnls
        return self._fallback_trade_pnls(trades)

    def _fallback_trade_pnls(self, trades: list[TradeRecord]) -> list[float]:
        """Infer trade P&L from portfolio-value deltas when no round trips exist."""

        executable = [trade for trade in trades if trade.action != TradeAction.HOLD]
        if len(executable) < 2:
            return []
        return [
            executable[index].portfolio_value - executable[index - 1].portfolio_value
            for index in range(1, len(executable))
        ]

    def _win_rate(self, trade_pnls: list[float]) -> float:
        """Calculate fraction of completed trades with positive P&L."""

        if not trade_pnls:
            return 0.0
        return len([value for value in trade_pnls if value > 0]) / len(trade_pnls)

    def _profit_factor(self, winning: list[float], losing: list[float]) -> float:
        """Calculate gross profit divided by gross loss."""

        gross_profit = sum(winning)
        gross_loss = abs(sum(losing))
        if gross_loss == 0:
            return gross_profit if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def _average_trade_duration_hours(self, trades: list[TradeRecord]) -> float:
        """Calculate average duration between long entry and exit in hours."""

        entry_by_symbol: dict[str, datetime] = {}
        durations: list[float] = []

        for trade in trades:
            if trade.action == TradeAction.BUY and trade.position_after > 0:
                entry_by_symbol.setdefault(trade.symbol, trade.timestamp)
                continue
            if trade.action == TradeAction.SELL and trade.symbol in entry_by_symbol:
                elapsed = trade.timestamp - entry_by_symbol.pop(trade.symbol)
                durations.append(elapsed.total_seconds() / 3_600)

        if not durations:
            return 0.0
        return float(np.mean(durations))

    def _exposure_time(
        self, trades: list[TradeRecord], equity_curve: list[EquityPoint]
    ) -> float:
        """Calculate fraction of elapsed time with at least one open position."""

        if len(equity_curve) < 2:
            return 0.0
        total_seconds = (
            equity_curve[-1].timestamp - equity_curve[0].timestamp
        ).total_seconds()
        if total_seconds <= 0:
            return 0.0

        trade_events = sorted(
            trades, key=lambda trade: (trade.timestamp, trade.trade_id)
        )
        position_by_symbol: dict[str, float] = defaultdict(float)
        event_index = 0
        exposed_seconds = 0.0

        for current, following in zip(equity_curve, equity_curve[1:]):
            while (
                event_index < len(trade_events)
                and trade_events[event_index].timestamp <= current.timestamp
            ):
                event = trade_events[event_index]
                position_by_symbol[event.symbol] = event.position_after
                event_index += 1
            if any(abs(quantity) > 1e-12 for quantity in position_by_symbol.values()):
                exposed_seconds += (
                    following.timestamp - current.timestamp
                ).total_seconds()

        return max(0.0, min(1.0, exposed_seconds / total_seconds))


def _finite(value: float) -> float:
    """Return finite floats only, replacing NaN and infinities with zero."""

    if math.isnan(value) or math.isinf(value):
        return 0.0
    return float(value)


__all__ = ["MetricsCalculator"]
