"""Request and response schemas for model backtesting endpoints."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from enum import Enum

from algotrading.api.schemas.data_sources import (
    TrainingDataRequest,
    normalize_backtest_data_request,
)
from pydantic import BaseModel, Field, model_validator


class BacktestStatus(str, Enum):
    """Lifecycle states for a stored backtest result."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TradeAction(str, Enum):
    """Supported trade action labels in backtest trade records."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class BacktestRequest(BaseModel):
    """Payload used to run a backtest against a trained model generation."""

    model_id: str = Field(min_length=1)
    generation_id: str = Field(min_length=1)
    start_date: date
    end_date: date
    initial_capital: float = Field(default=100_000.0, gt=0)
    symbols: list[str] | None = None
    include_transaction_costs: bool = True
    description: str | None = None

    @model_validator(mode="after")
    def validate_date_range(self) -> BacktestRequest:
        """Ensure the requested backtest date range is valid."""

        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        if self.symbols is not None and not self.symbols:
            raise ValueError("symbols cannot be empty when provided")
        return self


class TradeRecord(BaseModel):
    """One action emitted during a backtest simulation."""

    trade_id: int
    timestamp: datetime
    symbol: str
    action: TradeAction
    quantity: float
    price: float
    cost: float
    position_after: float
    portfolio_value: float
    signal_confidence: float | None = None


class PerformanceMetrics(BaseModel):
    """Comprehensive performance metrics calculated from a backtest result."""

    total_return: float
    total_return_dollars: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration_days: int
    volatility: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    average_trade_duration: float
    exposure_time: float


class EquityPoint(BaseModel):
    """Portfolio equity snapshot at a point in simulated time."""

    timestamp: datetime
    portfolio_value: float
    cash: float
    position_value: float
    drawdown: float


class BacktestResult(BaseModel):
    """Stored result for one backtest execution."""

    backtest_id: str
    model_id: str
    generation_id: str
    status: BacktestStatus
    created_at: datetime
    completed_at: datetime | None = None
    request: BacktestRequest
    metrics: PerformanceMetrics | None = None
    equity_curve: list[EquityPoint] = Field(default_factory=list)
    error_message: str | None = None


class BacktestComparison(BaseModel):
    """Side-by-side comparison across multiple backtest results."""

    backtests: list[BacktestResult]
    metric_comparison: dict[str, list[float | None]] = Field(default_factory=dict)
    best_by_metric: dict[str, str] = Field(default_factory=dict)


class BacktestComparisonRequest(BaseModel):
    """Request payload for comparing stored backtest results."""

    backtest_ids: list[str] = Field(min_length=2)


def build_backtest_data_request(
    request: BacktestRequest,
    model_data_config: Mapping[str, object] | None,
    overrides: Mapping[str, object] | None = None,
) -> TrainingDataRequest:
    """Build the canonical data request for a backtest execution."""

    return normalize_backtest_data_request(
        model_data_config,
        start_date=request.start_date,
        end_date=request.end_date,
        symbols=request.symbols,
        overrides=overrides,
    )


__all__ = [
    "BacktestStatus",
    "TradeAction",
    "BacktestRequest",
    "TradeRecord",
    "PerformanceMetrics",
    "EquityPoint",
    "BacktestResult",
    "BacktestComparison",
    "BacktestComparisonRequest",
    "build_backtest_data_request",
]
