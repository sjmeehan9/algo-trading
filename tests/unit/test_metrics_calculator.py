"""Unit tests for backtest performance metric calculations."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from algotrading.api.schemas.backtesting import EquityPoint, TradeAction, TradeRecord
from algotrading.api.services.metrics_calculator import MetricsCalculator


def _equity_point(day_offset: int, value: float) -> EquityPoint:
    """Build an equity point at a deterministic daily timestamp."""

    return EquityPoint(
        timestamp=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=day_offset),
        portfolio_value=value,
        cash=value,
        position_value=0.0,
        drawdown=0.0,
    )


def _trade(
    trade_id: int,
    day_offset: int,
    action: TradeAction,
    quantity: float,
    price: float,
    position_after: float,
    portfolio_value: float,
) -> TradeRecord:
    """Build a trade record with deterministic timestamps."""

    return TradeRecord(
        trade_id=trade_id,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=day_offset),
        symbol="AAPL",
        action=action,
        quantity=quantity,
        price=price,
        cost=0.0,
        position_after=position_after,
        portfolio_value=portfolio_value,
    )


def test_metrics_calculator_returns_expected_core_metrics() -> None:
    """Calculator should return return, drawdown, and trade stats."""

    equity_curve = [
        _equity_point(0, 100_000.0),
        _equity_point(1, 102_000.0),
        _equity_point(2, 101_000.0),
        _equity_point(3, 105_000.0),
    ]
    trades = [
        _trade(1, 0, TradeAction.BUY, 100.0, 100.0, 100.0, 100_000.0),
        _trade(2, 1, TradeAction.SELL, 100.0, 110.0, 0.0, 101_000.0),
        _trade(3, 2, TradeAction.BUY, 90.0, 111.0, 90.0, 101_000.0),
        _trade(4, 3, TradeAction.SELL, 90.0, 109.0, 0.0, 100_820.0),
    ]

    metrics = MetricsCalculator().calculate_all(
        trades=trades,
        equity_curve=equity_curve,
        initial_capital=100_000.0,
    )

    assert metrics.total_return == pytest.approx(0.05)
    assert metrics.total_return_dollars == pytest.approx(5_000.0)
    assert metrics.max_drawdown == pytest.approx(1_000.0 / 102_000.0)
    assert metrics.total_trades == 4
    assert metrics.winning_trades == 1
    assert metrics.losing_trades == 1
    assert metrics.win_rate == pytest.approx(0.5)
    assert metrics.profit_factor == pytest.approx(1_000.0 / 180.0)
    assert metrics.average_trade_duration == pytest.approx(24.0)
    assert 0.0 <= metrics.exposure_time <= 1.0


def test_metrics_calculator_handles_empty_inputs_with_finite_values() -> None:
    """Calculator should avoid NaN/inf metrics for empty backtest data."""

    metrics = MetricsCalculator().calculate_all(
        trades=[],
        equity_curve=[],
        initial_capital=100_000.0,
    )

    assert metrics.total_return == 0.0
    assert metrics.sharpe_ratio == 0.0
    assert metrics.sortino_ratio == 0.0
    assert metrics.max_drawdown == 0.0
    assert metrics.total_trades == 0
    assert metrics.exposure_time == 0.0


def test_metrics_calculator_rejects_invalid_initial_capital() -> None:
    """Initial capital must be positive for return calculations."""

    with pytest.raises(ValueError):
        MetricsCalculator().calculate_all([], [], 0.0)
