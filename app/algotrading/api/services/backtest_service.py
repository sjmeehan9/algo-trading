"""Backtest execution, persistence, retrieval, and comparison service."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from math import ceil
from pathlib import Path
from threading import RLock
from typing import Protocol, TypeVar, runtime_checkable
from uuid import uuid4

import pandas as pd
from algotrading.api.schemas.backtesting import (
    BacktestComparison,
    BacktestRequest,
    BacktestResult,
    BacktestStatus,
    EquityPoint,
    TradeAction,
    TradeRecord,
)
from algotrading.api.schemas.common import PaginatedResponse
from algotrading.api.schemas.models import ModelConfigResponse
from algotrading.api.services.metrics_calculator import MetricsCalculator
from algotrading.api.services.model_service import (
    GenerationNotFoundError,
    ModelService,
)
from algotrading.src.models.tracking import EvaluationMetrics, Generation
from fastapi import Request

logger = logging.getLogger(__name__)

ResponseItemT = TypeVar("ResponseItemT")


class BacktestServiceError(Exception):
    """Base exception for backtest service operations."""


class BacktestValidationError(BacktestServiceError):
    """Raised when a backtest request or comparison request is invalid."""


class BacktestNotFoundError(BacktestServiceError):
    """Raised when a requested backtest result does not exist."""


class BacktestExecutionError(BacktestServiceError):
    """Raised by executors when a backtest cannot be executed."""


@dataclass(slots=True)
class BacktestExecutionResult:
    """Raw execution output returned by backtest executors."""

    trades: list[TradeRecord]
    equity_curve: list[EquityPoint]


@dataclass(slots=True)
class BacktestJobContext:
    """Execution context passed to a `BacktestExecutor`."""

    request: BacktestRequest
    model: ModelConfigResponse
    generation: Generation


@runtime_checkable
class BacktestExecutor(Protocol):
    """Protocol implemented by concrete backtest executors."""

    def execute(self, context: BacktestJobContext) -> BacktestExecutionResult:
        """Run a backtest and return trades plus equity snapshots."""


class DefaultBacktestExecutor:
    """CSV-backed executor for API backtests over historical market data.

    The executor uses market CSV files referenced by the model's
    ``training_data_config``. It provides a deterministic long-only simulation
    suitable for API evaluation workflows when a specialized model-specific
    executor has not been injected.
    """

    TRANSACTION_COST_RATE = 0.0005
    POSITION_CASH_FRACTION = 0.95
    DATA_PATH_KEYS = (
        "market_data_path",
        "backtest_data_path",
        "csv_path",
        "file_path",
        "data_path",
    )
    TIMESTAMP_COLUMNS = ("timestamp", "date", "datetime", "time")
    CLOSE_COLUMNS = ("close", "Close", "price", "last")

    def __init__(self, data_root: str | Path) -> None:
        """Initialize the executor with a project data root.

        Args:
            data_root: Base path used to resolve relative CSV paths.
        """

        self._data_root = Path(data_root)

    def execute(self, context: BacktestJobContext) -> BacktestExecutionResult:
        """Execute a file-backed deterministic backtest."""

        symbols = self._resolve_symbols(context.request, context.model)
        if not symbols:
            raise BacktestExecutionError(
                "No symbols were supplied in the request or model training_data_config"
            )

        allocation = context.request.initial_capital / len(symbols)
        all_trades: list[TradeRecord] = []
        curves: list[list[EquityPoint]] = []
        next_trade_id = 1

        for symbol in symbols:
            frame = self._load_market_data(
                symbol=symbol,
                config=context.model.training_data_config,
                start_date=context.request.start_date,
                end_date=context.request.end_date,
            )
            trades, curve, next_trade_id = self._simulate_symbol(
                frame=frame,
                symbol=symbol,
                initial_capital=allocation,
                include_costs=context.request.include_transaction_costs,
                starting_trade_id=next_trade_id,
            )
            all_trades.extend(trades)
            curves.append(curve)

        equity_curve = self._aggregate_equity_curves(
            curves=curves,
            initial_capital=context.request.initial_capital,
        )
        if not equity_curve:
            raise BacktestExecutionError("Backtest produced no equity curve")
        return BacktestExecutionResult(trades=all_trades, equity_curve=equity_curve)

    def _resolve_symbols(
        self, request: BacktestRequest, model: ModelConfigResponse
    ) -> list[str]:
        """Resolve requested symbols from explicit request or model config."""

        raw_symbols: object = request.symbols
        if raw_symbols is None:
            raw_symbols = model.training_data_config.get("symbols")
        if raw_symbols is None:
            raw_symbols = model.training_data_config.get("symbol")

        if isinstance(raw_symbols, str):
            return [raw_symbols.strip().upper()] if raw_symbols.strip() else []
        if isinstance(raw_symbols, list):
            return [
                str(item).strip().upper() for item in raw_symbols if str(item).strip()
            ]
        return []

    def _load_market_data(
        self,
        *,
        symbol: str,
        config: dict[str, object],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Load and normalize CSV market data for one symbol."""

        for candidate in self._market_data_candidates(symbol=symbol, config=config):
            if not candidate.exists() or not candidate.is_file():
                continue
            frame = pd.read_csv(candidate)
            normalized = self._normalize_market_frame(frame, start_date, end_date)
            if not normalized.empty:
                return normalized

        raise BacktestExecutionError(
            f"No market data CSV found for symbol '{symbol}' in configured paths"
        )

    def _market_data_candidates(
        self, *, symbol: str, config: dict[str, object]
    ) -> list[Path]:
        """Return possible CSV files for a symbol based on training data config."""

        candidates: list[Path] = []

        symbol_files = config.get("symbol_files")
        if isinstance(symbol_files, dict):
            specific = symbol_files.get(symbol) or symbol_files.get(symbol.upper())
            if isinstance(specific, str):
                candidates.append(self._resolve_path(specific))

        data_files = config.get("data_files")
        if isinstance(data_files, dict):
            specific = data_files.get(symbol) or data_files.get(symbol.upper())
            if isinstance(specific, str):
                candidates.append(self._resolve_path(specific))
        elif isinstance(data_files, list):
            candidates.extend(
                self._resolve_path(str(item)) for item in data_files if str(item)
            )

        for key in self.DATA_PATH_KEYS:
            value = config.get(key)
            if isinstance(value, str) and value:
                candidates.append(self._resolve_path(value))

        default_saved_data = self._data_root / "data" / "saved_data"
        if default_saved_data.exists():
            candidates.append(default_saved_data)

        expanded: list[Path] = []
        for candidate in candidates:
            if candidate.is_dir():
                expanded.extend(sorted(candidate.rglob(f"{symbol}*.csv")))
                expanded.extend(sorted(candidate.rglob(f"*{symbol}*.csv")))
                expanded.extend(sorted(candidate.rglob("*.csv")))
            else:
                expanded.append(candidate)

        unique: dict[str, Path] = {}
        for item in expanded:
            unique[str(item)] = item
        return list(unique.values())

    def _resolve_path(self, raw_path: str) -> Path:
        """Resolve configured absolute or project-relative paths."""

        path = Path(raw_path).expanduser()
        if path.is_absolute():
            return path
        return self._data_root / path

    def _normalize_market_frame(
        self, frame: pd.DataFrame, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """Normalize raw CSV market data to timestamp and close columns."""

        timestamp_column = self._first_existing_column(frame, self.TIMESTAMP_COLUMNS)
        close_column = self._first_existing_column(frame, self.CLOSE_COLUMNS)
        if timestamp_column is None or close_column is None:
            return pd.DataFrame(columns=["timestamp", "close"])

        normalized = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    frame[timestamp_column], utc=True, errors="coerce"
                ),
                "close": pd.to_numeric(frame[close_column], errors="coerce"),
            }
        )
        normalized = normalized.dropna(subset=["timestamp", "close"])
        start_dt = datetime.combine(start_date, time.min, tzinfo=UTC)
        end_dt = datetime.combine(end_date, time.max, tzinfo=UTC)
        normalized = normalized[
            (normalized["timestamp"] >= start_dt) & (normalized["timestamp"] <= end_dt)
        ]
        normalized = normalized.sort_values("timestamp")
        return normalized.reset_index(drop=True)

    def _first_existing_column(
        self, frame: pd.DataFrame, candidates: tuple[str, ...]
    ) -> str | None:
        """Return the first column whose name matches any candidate."""

        column_by_lower = {str(column).lower(): str(column) for column in frame.columns}
        for candidate in candidates:
            match = column_by_lower.get(candidate.lower())
            if match is not None:
                return match
        return None

    def _simulate_symbol(
        self,
        *,
        frame: pd.DataFrame,
        symbol: str,
        initial_capital: float,
        include_costs: bool,
        starting_trade_id: int,
    ) -> tuple[list[TradeRecord], list[EquityPoint], int]:
        """Run a deterministic long-only price-momentum simulation for one symbol."""

        cash = initial_capital
        quantity = 0.0
        previous_price: float | None = None
        trade_id = starting_trade_id
        trades: list[TradeRecord] = []
        curve: list[EquityPoint] = []

        for row in frame.itertuples(index=False):
            timestamp = row.timestamp.to_pydatetime()
            price = float(row.close)
            action = TradeAction.HOLD
            traded_quantity = 0.0
            cost = 0.0

            if previous_price is not None and price > previous_price and quantity == 0:
                notional_budget = cash * self.POSITION_CASH_FRACTION
                traded_quantity = notional_budget / max(price, 1e-12)
                notional = traded_quantity * price
                cost = self._transaction_cost(notional, include_costs)
                if notional + cost <= cash and traded_quantity > 0:
                    cash -= notional + cost
                    quantity += traded_quantity
                    action = TradeAction.BUY
            elif previous_price is not None and price < previous_price and quantity > 0:
                traded_quantity = quantity
                notional = traded_quantity * price
                cost = self._transaction_cost(notional, include_costs)
                cash += notional - cost
                quantity = 0.0
                action = TradeAction.SELL

            portfolio_value = cash + (quantity * price)
            if action != TradeAction.HOLD:
                trades.append(
                    TradeRecord(
                        trade_id=trade_id,
                        timestamp=timestamp,
                        symbol=symbol,
                        action=action,
                        quantity=traded_quantity,
                        price=price,
                        cost=cost,
                        position_after=quantity,
                        portfolio_value=portfolio_value,
                    )
                )
                trade_id += 1

            curve.append(
                EquityPoint(
                    timestamp=timestamp,
                    portfolio_value=portfolio_value,
                    cash=cash,
                    position_value=quantity * price,
                    drawdown=0.0,
                )
            )
            previous_price = price

        if quantity > 0 and not frame.empty:
            final_row = frame.iloc[-1]
            timestamp = final_row["timestamp"].to_pydatetime()
            price = float(final_row["close"])
            traded_quantity = quantity
            notional = traded_quantity * price
            cost = self._transaction_cost(notional, include_costs)
            cash += notional - cost
            quantity = 0.0
            portfolio_value = cash
            trades.append(
                TradeRecord(
                    trade_id=trade_id,
                    timestamp=timestamp,
                    symbol=symbol,
                    action=TradeAction.SELL,
                    quantity=traded_quantity,
                    price=price,
                    cost=cost,
                    position_after=quantity,
                    portfolio_value=portfolio_value,
                )
            )
            trade_id += 1
            if curve:
                curve[-1] = EquityPoint(
                    timestamp=timestamp,
                    portfolio_value=portfolio_value,
                    cash=cash,
                    position_value=0.0,
                    drawdown=0.0,
                )

        return trades, curve, trade_id

    def _transaction_cost(self, notional: float, include_costs: bool) -> float:
        """Calculate transaction cost for a notional trade value."""

        if not include_costs:
            return 0.0
        return abs(notional) * self.TRANSACTION_COST_RATE

    def _aggregate_equity_curves(
        self, *, curves: list[list[EquityPoint]], initial_capital: float
    ) -> list[EquityPoint]:
        """Aggregate per-symbol curves into a total portfolio equity curve."""

        frames: list[pd.DataFrame] = []
        for curve in curves:
            if not curve:
                continue
            frame = pd.DataFrame(
                [
                    {
                        "timestamp": point.timestamp,
                        "portfolio_value": point.portfolio_value,
                        "cash": point.cash,
                        "position_value": point.position_value,
                    }
                    for point in curve
                ]
            ).set_index("timestamp")
            frames.append(frame.sort_index())

        if not frames:
            return []

        union_index = frames[0].index
        for frame in frames[1:]:
            union_index = union_index.union(frame.index)
        union_index = union_index.sort_values()

        summed = pd.DataFrame(index=union_index)
        summed["portfolio_value"] = 0.0
        summed["cash"] = 0.0
        summed["position_value"] = 0.0
        for frame in frames:
            aligned = frame.reindex(union_index).ffill().bfill()
            summed = summed.add(aligned, fill_value=0.0)

        peak = initial_capital
        points: list[EquityPoint] = []
        for timestamp, row in summed.iterrows():
            portfolio_value = float(row["portfolio_value"])
            peak = max(peak, portfolio_value)
            drawdown = 0.0 if peak <= 0 else (portfolio_value - peak) / peak
            points.append(
                EquityPoint(
                    timestamp=timestamp.to_pydatetime(),
                    portfolio_value=portfolio_value,
                    cash=float(row["cash"]),
                    position_value=float(row["position_value"]),
                    drawdown=drawdown,
                )
            )
        return points


class BacktestService:
    """Coordinate backtest execution, persistence, and comparison."""

    COMPARISON_METRICS = (
        "total_return",
        "total_return_dollars",
        "annualized_return",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "volatility",
        "win_rate",
        "profit_factor",
        "total_trades",
    )
    LOWER_IS_BETTER = {"max_drawdown", "volatility"}

    def __init__(
        self,
        model_service: ModelService,
        metrics_calculator: MetricsCalculator,
        executor: BacktestExecutor,
        results_path: str | Path | None = None,
    ) -> None:
        """Initialize service dependencies.

        Args:
            model_service: Model service used to validate models and generations.
            metrics_calculator: Calculator used to derive result metrics.
            executor: Backtest executor implementation.
            results_path: Optional JSON path used for result persistence.
        """

        self.model_service = model_service
        self.generation_tracker = model_service.generation_tracker
        self.metrics_calculator = metrics_calculator
        self.executor = executor
        self._results_path = Path(results_path) if results_path else None
        self._lock = RLock()
        self._results: dict[str, BacktestResult] = {}
        self._trades: dict[str, list[TradeRecord]] = {}
        self._load_results()

    async def run_backtest(self, request: BacktestRequest) -> BacktestResult:
        """Run a backtest and persist the resulting metrics and trades."""

        model, generation = self._resolve_context(request)
        backtest_id = f"backtest-{uuid4().hex[:12]}"
        started = BacktestResult(
            backtest_id=backtest_id,
            model_id=model.model_id,
            generation_id=generation.generation_id,
            status=BacktestStatus.RUNNING,
            created_at=datetime.now(tz=UTC),
            request=request,
        )

        with self._lock:
            self._results[backtest_id] = started
            self._trades[backtest_id] = []
            self._persist_results_locked()

        context = BacktestJobContext(
            request=request,
            model=model,
            generation=generation,
        )

        try:
            execution = await asyncio.to_thread(self.executor.execute, context)
            metrics = self.metrics_calculator.calculate_all(
                trades=execution.trades,
                equity_curve=execution.equity_curve,
                initial_capital=request.initial_capital,
            )
            completed = started.model_copy(
                update={
                    "status": BacktestStatus.COMPLETED,
                    "completed_at": datetime.now(tz=UTC),
                    "metrics": metrics,
                    "equity_curve": execution.equity_curve,
                }
            )
            with self._lock:
                self._results[backtest_id] = completed
                self._trades[backtest_id] = execution.trades
                self._persist_results_locked()
            self._attach_generation_evaluation(generation.generation_id, metrics)
            return completed
        except Exception as exc:
            logger.exception("Backtest %s failed", backtest_id)
            failed = started.model_copy(
                update={
                    "status": BacktestStatus.FAILED,
                    "completed_at": datetime.now(tz=UTC),
                    "error_message": str(exc),
                }
            )
            with self._lock:
                self._results[backtest_id] = failed
                self._trades[backtest_id] = []
                self._persist_results_locked()
            return failed

    def list_results(
        self,
        *,
        model_id: str | None = None,
        generation_id: str | None = None,
        result_status: BacktestStatus | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> PaginatedResponse[BacktestResult]:
        """List backtest results with optional filters and pagination."""

        with self._lock:
            results = list(self._results.values())
        if model_id is not None:
            results = [result for result in results if result.model_id == model_id]
        if generation_id is not None:
            results = [
                result for result in results if result.generation_id == generation_id
            ]
        if result_status is not None:
            results = [result for result in results if result.status == result_status]
        results.sort(key=lambda result: result.created_at, reverse=True)
        return _paginate(results, page=page, page_size=page_size)

    def get_result(self, backtest_id: str) -> BacktestResult:
        """Return one stored backtest result by ID."""

        with self._lock:
            result = self._results.get(backtest_id)
        if result is None:
            raise BacktestNotFoundError(f"Backtest '{backtest_id}' not found")
        return result

    def get_trades(self, backtest_id: str) -> list[TradeRecord]:
        """Return trade records for a stored backtest."""

        self.get_result(backtest_id)
        with self._lock:
            trades = list(self._trades.get(backtest_id, []))
        return trades

    def compare_backtests(self, backtest_ids: list[str]) -> BacktestComparison:
        """Compare multiple stored backtests across core performance metrics."""

        unique_ids = list(dict.fromkeys(backtest_ids))
        if len(unique_ids) < 2:
            raise BacktestValidationError("At least two backtest_ids are required")

        results = [self.get_result(backtest_id) for backtest_id in unique_ids]
        metric_comparison: dict[str, list[float | None]] = {}
        best_by_metric: dict[str, str] = {}

        for metric_name in self.COMPARISON_METRICS:
            values = [
                (
                    _metric_value(result, metric_name)
                    if result.metrics is not None
                    and result.status == BacktestStatus.COMPLETED
                    else None
                )
                for result in results
            ]
            metric_comparison[metric_name] = values

            indexed = [
                (index, value)
                for index, value in enumerate(values)
                if value is not None
            ]
            if not indexed:
                continue
            if metric_name in self.LOWER_IS_BETTER:
                best_index = min(indexed, key=lambda item: item[1])[0]
            else:
                best_index = max(indexed, key=lambda item: item[1])[0]
            best_by_metric[metric_name] = results[best_index].backtest_id

        return BacktestComparison(
            backtests=results,
            metric_comparison=metric_comparison,
            best_by_metric=best_by_metric,
        )

    def _resolve_context(
        self, request: BacktestRequest
    ) -> tuple[ModelConfigResponse, Generation]:
        """Resolve and validate model plus generation for a backtest."""

        model = self.model_service.get_model(request.model_id)
        generation = self.generation_tracker.get_generation(request.generation_id)
        if generation is None or generation.model_id != model.model_id:
            raise GenerationNotFoundError(
                f"Generation '{request.generation_id}' not found for model '{model.model_id}'"
            )
        if generation.status not in {"completed", "evaluated"}:
            raise BacktestValidationError(
                f"Generation '{generation.generation_id}' must be completed or evaluated before backtesting"
            )
        return model, generation

    def _attach_generation_evaluation(
        self, generation_id: str, metrics: object
    ) -> None:
        """Attach backtest performance metrics to generation history."""

        if not hasattr(metrics, "sharpe_ratio"):
            return
        try:
            self.generation_tracker.add_evaluation(
                generation_id,
                EvaluationMetrics(
                    sharpe_ratio=metrics.sharpe_ratio,
                    max_drawdown=metrics.max_drawdown,
                    total_return=metrics.total_return,
                    win_rate=metrics.win_rate,
                    profit_factor=metrics.profit_factor,
                    num_trades=metrics.total_trades,
                    custom_metrics={
                        "annualized_return": metrics.annualized_return,
                        "sortino_ratio": metrics.sortino_ratio,
                        "volatility": metrics.volatility,
                    },
                ),
            )
        except Exception:
            logger.exception(
                "Failed to attach backtest evaluation metrics to generation %s",
                generation_id,
            )

    def _persist_results_locked(self) -> None:
        """Persist current results and trades while holding the service lock."""

        if self._results_path is None:
            return
        self._results_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "saved_at": datetime.now(tz=UTC).isoformat(),
            "results": [
                result.model_dump(mode="json") for result in self._results.values()
            ],
            "trades": {
                backtest_id: [trade.model_dump(mode="json") for trade in trades]
                for backtest_id, trades in self._trades.items()
            },
        }
        self._results_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _load_results(self) -> None:
        """Load persisted backtest results if a store already exists."""

        if self._results_path is None or not self._results_path.exists():
            return
        try:
            payload = json.loads(self._results_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning(
                "Could not parse backtest results file %s", self._results_path
            )
            return

        loaded_results: dict[str, BacktestResult] = {}
        loaded_trades: dict[str, list[TradeRecord]] = {}

        for item in payload.get("results", []):
            try:
                result = BacktestResult.model_validate(item)
            except Exception:
                logger.exception("Skipping invalid persisted backtest result entry")
                continue
            loaded_results[result.backtest_id] = result

        raw_trades = payload.get("trades", {})
        if isinstance(raw_trades, dict):
            for backtest_id, entries in raw_trades.items():
                if not isinstance(entries, list):
                    continue
                parsed: list[TradeRecord] = []
                for entry in entries:
                    try:
                        parsed.append(TradeRecord.model_validate(entry))
                    except Exception:
                        logger.exception(
                            "Skipping invalid trade entry for backtest %s",
                            backtest_id,
                        )
                loaded_trades[str(backtest_id)] = parsed

        with self._lock:
            self._results = loaded_results
            self._trades = loaded_trades


def _metric_value(result: BacktestResult, metric_name: str) -> float | None:
    """Extract one numeric metric from a result when available."""

    if result.metrics is None:
        return None
    value = getattr(result.metrics, metric_name, None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _paginate(
    items: list[ResponseItemT],
    page: int,
    page_size: int,
) -> PaginatedResponse[ResponseItemT]:
    """Return a consistent paginated response object."""

    if page < 1:
        raise BacktestValidationError("page must be greater than or equal to 1")
    if page_size < 1:
        raise BacktestValidationError("page_size must be greater than or equal to 1")

    total = len(items)
    pages = ceil(total / page_size) if total > 0 else 0
    start = (page - 1) * page_size
    end = start + page_size
    if page > 1 and start >= total and total > 0:
        raise BacktestValidationError(
            f"Requested page {page} exceeds available pages ({pages})"
        )
    return PaginatedResponse[ResponseItemT](
        items=items[start:end],
        total=total,
        page=page,
        page_size=page_size,
        pages=pages,
    )


def create_default_backtest_service(
    *,
    model_service: ModelService,
    project_root: Path | None = None,
    executor: BacktestExecutor | None = None,
) -> BacktestService:
    """Build a BacktestService with filesystem-backed defaults."""

    root = project_root or Path(__file__).resolve().parents[4]
    data_dir = root / "data" / "api"
    results_path = data_dir / "backtest_results.json"
    resolved_executor = executor or DefaultBacktestExecutor(data_root=root)
    return BacktestService(
        model_service=model_service,
        metrics_calculator=MetricsCalculator(),
        executor=resolved_executor,
        results_path=results_path,
    )


def get_backtest_service(request: Request) -> BacktestService:
    """FastAPI dependency resolver for the shared BacktestService instance."""

    service = getattr(request.app.state, "backtest_service", None)
    if service is None:
        from algotrading.api.services.model_service import get_model_service

        model_service = get_model_service(request)
        service = create_default_backtest_service(model_service=model_service)
        request.app.state.backtest_service = service
    return service


__all__ = [
    "BacktestExecutor",
    "BacktestExecutionResult",
    "BacktestJobContext",
    "BacktestService",
    "BacktestServiceError",
    "BacktestValidationError",
    "BacktestNotFoundError",
    "BacktestExecutionError",
    "DefaultBacktestExecutor",
    "create_default_backtest_service",
    "get_backtest_service",
]
