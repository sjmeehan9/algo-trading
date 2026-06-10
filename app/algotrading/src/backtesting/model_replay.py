"""Model-driven backtesting replay (Phase 7 Component 7.5).

This module replaces the model-independent momentum simulation that the API
backtesting path historically used with a replay executor that loads the
selected trained generation and drives trade actions from the model's own
predictions over canonical sourced market data.

The runtime flow is:

1. ``ModelDrivenBacktestExecutor.execute(...)`` validates that the selected
   generation is completed and has a readable ``model_path``.
2. It resolves the canonical data request through
   :func:`build_backtest_data_request` (the same contract training uses),
   then ensures market data — and news when the model requires it — exists in
   the Phase 7 local store via :class:`DataAcquisitionService`.
3. :class:`BacktestReplayEngine` builds a single-pass ``TradingEnv`` from the
   sourced bars (reusing the Component 7.4 environment factory), loads the SB3
   model artifact, and walks the bars once. At each bar the model predicts an
   action (0 = hold, 1 = buy, 2 = sell) and a deterministic fill model updates
   cash, position, trades, and the equity curve.

The replay emits ``TradeRecord`` and ``EquityPoint`` values that conform to the
existing backtesting schemas, so downstream metrics, persistence, and
generation-evaluation wiring are unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from algotrading.api.schemas.backtesting import (
    EquityPoint,
    TradeAction,
    TradeRecord,
    build_backtest_data_request,
)
from algotrading.api.schemas.data_sources import (
    TrainingDataRequest,
    normalize_training_data_request,
)
from algotrading.api.schemas.models import ModelConfigResponse
from algotrading.api.services.data_acquisition_service import DataAcquisitionService
from algotrading.api.services.model_service import ModelService
from algotrading.api.training.factories import (
    BuiltTradingEnvironment,
    TrainingFactoryError,
    build_trading_environment,
)
from algotrading.src.envs.trading_env import TradingEnv
from algotrading.src.trainers.exceptions import ModelLoadError
from algotrading.src.trainers.sb3_trainer import SB3Algorithm, StableBaselines3Trainer

logger = logging.getLogger(__name__)

# Discrete action indices emitted by the trained policy / TradingEnv action space.
_HOLD_ACTION = 0
_BUY_ACTION = 1
_SELL_ACTION = 2

# Default per-trade transaction cost rate used when the request enables costs but
# the model/job data config does not override it. Mirrors the historical default.
_DEFAULT_TRANSACTION_COST_RATE = 0.0005


class BacktestReplayError(Exception):
    """Raised when a model-driven backtest cannot be replayed."""


@dataclass(slots=True)
class ReplayResult:
    """Internal per-symbol replay output before portfolio aggregation."""

    trades: list[TradeRecord]
    equity_curve: list[EquityPoint]


class ReplayFillModel:
    """Deterministic long-only fill model for historical replay.

    The fill model is intentionally simple and deterministic so that backtests
    are reproducible and auditable:

    * A ``BUY`` action with no open position invests an even fraction of the
      available cash at the current bar's close price.
    * A ``SELL`` action with an open position liquidates the entire position at
      the current bar's close price.
    * A ``HOLD`` action (or an action the portfolio cannot afford) leaves the
      position unchanged.
    * Transaction costs are applied as a fraction of traded notional when the
      backtest request enables them.
    * Cash can never go below zero and the position can never go below zero;
      shorting is intentionally out of scope until explicitly added.
    """

    def __init__(
        self,
        *,
        initial_capital: float,
        include_costs: bool,
        transaction_cost_rate: float = _DEFAULT_TRANSACTION_COST_RATE,
        position_cash_fraction: float = 1.0,
    ) -> None:
        """Initialize the fill model with capital and cost settings.

        Args:
            initial_capital: Starting cash allocated to this symbol.
            include_costs: Whether to charge transaction costs on fills.
            transaction_cost_rate: Cost charged as a fraction of traded notional.
            position_cash_fraction: Fraction of available cash deployed on a buy.
        """

        if initial_capital <= 0:
            raise BacktestReplayError("initial_capital must be positive")
        self._cash = float(initial_capital)
        self._quantity = 0.0
        self._include_costs = bool(include_costs)
        self._transaction_cost_rate = max(0.0, float(transaction_cost_rate))
        self._position_cash_fraction = min(1.0, max(0.0, float(position_cash_fraction)))

    @property
    def cash(self) -> float:
        """Current available cash."""

        return self._cash

    @property
    def quantity(self) -> float:
        """Current open position quantity (shares/units)."""

        return self._quantity

    def portfolio_value(self, price: float) -> float:
        """Return cash plus marked-to-market position value at ``price``."""

        return self._cash + (self._quantity * price)

    def apply(
        self, action: TradeAction, price: float
    ) -> tuple[TradeAction, float, float]:
        """Apply one model action at ``price`` and return the executed fill.

        Returns:
            A tuple of ``(executed_action, traded_quantity, cost)``. The
            executed action collapses to ``HOLD`` when the requested action
            cannot be filled (e.g. a buy with no cash, or a sell with no
            position).
        """

        price = float(price)
        if price <= 0:
            return TradeAction.HOLD, 0.0, 0.0

        if action == TradeAction.BUY and self._quantity == 0.0:
            return self._open_long(price)
        if action == TradeAction.SELL and self._quantity > 0.0:
            return self._close_long(price)
        return TradeAction.HOLD, 0.0, 0.0

    def liquidate(self, price: float) -> tuple[TradeAction, float, float]:
        """Force-close any open position at ``price`` (end-of-replay flatten)."""

        if self._quantity <= 0.0:
            return TradeAction.HOLD, 0.0, 0.0
        return self._close_long(float(price))

    def _open_long(self, price: float) -> tuple[TradeAction, float, float]:
        budget = self._cash * self._position_cash_fraction
        if budget <= 0:
            return TradeAction.HOLD, 0.0, 0.0
        # Solve quantity so that notional + cost never exceeds the budget.
        cost_multiplier = 1.0 + (
            self._transaction_cost_rate if self._include_costs else 0.0
        )
        quantity = budget / (price * cost_multiplier)
        if quantity <= 0:
            return TradeAction.HOLD, 0.0, 0.0
        notional = quantity * price
        cost = self._transaction_cost(notional)
        total = notional + cost
        # Allow a tiny floating-point tolerance so a full-cash buy is not
        # rejected by rounding noise; clamp cash to zero afterwards.
        if total > self._cash * (1.0 + 1e-9):
            return TradeAction.HOLD, 0.0, 0.0
        self._cash = max(0.0, self._cash - total)
        self._quantity += quantity
        return TradeAction.BUY, quantity, cost

    def _close_long(self, price: float) -> tuple[TradeAction, float, float]:
        quantity = self._quantity
        notional = quantity * price
        cost = self._transaction_cost(notional)
        self._cash += notional - cost
        if self._cash < 0:
            self._cash = 0.0
        self._quantity = 0.0
        return TradeAction.SELL, quantity, cost

    def _transaction_cost(self, notional: float) -> float:
        if not self._include_costs:
            return 0.0
        return abs(notional) * self._transaction_cost_rate


class BacktestReplayEngine:
    """Replay one trained generation over sourced bars to produce trades.

    The engine builds a single-pass ``TradingEnv`` from canonical market data,
    loads the SB3 artifact referenced by the generation, and steps through the
    environment once. The model selects an action per bar; the deterministic
    :class:`ReplayFillModel` translates that action into cash/position changes
    and emits schema-conformant ``TradeRecord`` and ``EquityPoint`` values.
    """

    def __init__(
        self,
        *,
        data_service: DataAcquisitionService,
        model_service: ModelService,
        project_root: Path | None = None,
        deterministic: bool = True,
    ) -> None:
        """Initialize the replay engine.

        Args:
            data_service: Phase 7 acquisition service for canonical market/news.
            model_service: Model service used for supporting-model validation.
            project_root: Optional override for the project data root.
            deterministic: Whether to use deterministic model predictions.
        """

        self._data_service = data_service
        self._model_service = model_service
        self._project_root = project_root
        self._deterministic = deterministic

    def run(
        self,
        *,
        model: ModelConfigResponse,
        model_path: str | Path,
        request: TrainingDataRequest,
        initial_capital: float,
        include_costs: bool,
    ) -> ReplayResult:
        """Replay the model for one symbol and return trades plus equity curve.

        Args:
            model: The core RL model whose config shapes the environment.
            model_path: Filesystem path to the trained SB3 artifact.
            request: Canonical data request for the symbol/date range.
            initial_capital: Capital allocated to this symbol.
            include_costs: Whether to charge transaction costs.

        Returns:
            The per-symbol replay output.

        Raises:
            BacktestReplayError: If the environment cannot be built or the model
                cannot be loaded.
        """

        built = self._build_environment(model=model, request=request)
        trainer = self._load_trainer(model=model, model_path=model_path, built=built)
        symbol = built.symbol
        fill_model = ReplayFillModel(
            initial_capital=initial_capital,
            include_costs=include_costs,
            transaction_cost_rate=self._transaction_cost_rate(request, include_costs),
        )
        return self._replay(
            env=built.env,
            trainer=trainer,
            symbol=symbol,
            fill_model=fill_model,
        )

    def _build_environment(
        self,
        *,
        model: ModelConfigResponse,
        request: TrainingDataRequest,
    ) -> BuiltTradingEnvironment:
        try:
            return build_backtest_environment(
                model=model,
                request=request,
                data_service=self._data_service,
                model_service=self._model_service,
                project_root=self._project_root,
            )
        except TrainingFactoryError as exc:
            raise BacktestReplayError(
                f"Could not build backtest environment for model '{model.model_id}': "
                f"{exc}"
            ) from exc

    def _load_trainer(
        self,
        *,
        model: ModelConfigResponse,
        model_path: str | Path,
        built: BuiltTradingEnvironment,
    ) -> StableBaselines3Trainer:
        algorithm = _resolve_sb3_algorithm(model)
        policy = _resolve_policy(model=model, env=built.env)
        trainer = StableBaselines3Trainer(algorithm=algorithm, policy=policy)
        try:
            trainer.load(str(model_path), env=built.env)
        except ModelLoadError as exc:
            raise BacktestReplayError(
                f"Could not load trained model artifact at '{model_path}' for "
                f"model '{model.model_id}': {exc}"
            ) from exc
        if not trainer.is_trained:
            raise BacktestReplayError(
                f"Loaded model artifact at '{model_path}' is not usable for replay"
            )
        return trainer

    def _replay(
        self,
        *,
        env: TradingEnv,
        trainer: StableBaselines3Trainer,
        symbol: str,
        fill_model: ReplayFillModel,
    ) -> ReplayResult:
        observation, _info = env.reset()
        trades: list[TradeRecord] = []
        curve: list[EquityPoint] = []
        trade_id = 1
        peak = fill_model.portfolio_value(self._current_close(env))

        terminated = False
        truncated = False
        # Walk the env once. Each iteration processes the *current* bar: the
        # model picks an action, we fill it at this bar's close, record equity,
        # then advance the env to expose the next bar (or terminate).
        guard = len(env.state_builder.final_dataframe) + 2
        steps = 0
        while not (terminated or truncated) and steps <= guard:
            timestamp = self._current_timestamp(env)
            price = self._current_close(env)
            action_index, _ = trainer.predict(
                observation, deterministic=self._deterministic
            )
            requested_action = _action_from_index(action_index)
            executed, quantity, cost = fill_model.apply(requested_action, price)
            portfolio_value = fill_model.portfolio_value(price)

            if executed != TradeAction.HOLD:
                trades.append(
                    TradeRecord(
                        trade_id=trade_id,
                        timestamp=timestamp,
                        symbol=symbol,
                        action=executed,
                        quantity=quantity,
                        price=price,
                        cost=cost,
                        position_after=fill_model.quantity,
                        portfolio_value=portfolio_value,
                    )
                )
                trade_id += 1

            peak = max(peak, portfolio_value)
            drawdown = 0.0 if peak <= 0 else (portfolio_value - peak) / peak
            curve.append(
                EquityPoint(
                    timestamp=timestamp,
                    portfolio_value=portfolio_value,
                    cash=fill_model.cash,
                    position_value=fill_model.quantity * price,
                    drawdown=drawdown,
                )
            )

            observation, _reward, terminated, truncated, _step_info = env.step(
                int(_action_to_env_index(executed))
            )
            steps += 1

        self._flatten_final_position(
            env=env,
            fill_model=fill_model,
            symbol=symbol,
            trades=trades,
            curve=curve,
            next_trade_id=trade_id,
            peak=peak,
        )

        if not curve:
            raise BacktestReplayError(
                f"Replay for symbol '{symbol}' produced no equity points"
            )
        return ReplayResult(trades=trades, equity_curve=curve)

    def _flatten_final_position(
        self,
        *,
        env: TradingEnv,
        fill_model: ReplayFillModel,
        symbol: str,
        trades: list[TradeRecord],
        curve: list[EquityPoint],
        next_trade_id: int,
        peak: float,
    ) -> None:
        """Liquidate any residual position at the final bar's close price."""

        if fill_model.quantity <= 0.0:
            return

        timestamp = self._current_timestamp(env)
        price = self._current_close(env)
        executed, quantity, cost = fill_model.liquidate(price)
        if executed == TradeAction.HOLD:
            return

        portfolio_value = fill_model.portfolio_value(price)
        trades.append(
            TradeRecord(
                trade_id=next_trade_id,
                timestamp=timestamp,
                symbol=symbol,
                action=executed,
                quantity=quantity,
                price=price,
                cost=cost,
                position_after=fill_model.quantity,
                portfolio_value=portfolio_value,
            )
        )
        peak = max(peak, portfolio_value)
        drawdown = 0.0 if peak <= 0 else (portfolio_value - peak) / peak
        final_point = EquityPoint(
            timestamp=timestamp,
            portfolio_value=portfolio_value,
            cash=fill_model.cash,
            position_value=0.0,
            drawdown=drawdown,
        )
        if curve and curve[-1].timestamp == timestamp:
            curve[-1] = final_point
        else:
            curve.append(final_point)

    def _current_timestamp(self, env: TradingEnv) -> datetime:
        state_df = getattr(env.state_builder, "state_df", None)
        if state_df is None or state_df.empty or "date" not in state_df:
            return datetime.now(tz=UTC)
        raw = state_df["date"].iloc[-1]
        return _to_utc_datetime(raw)

    def _current_close(self, env: TradingEnv) -> float:
        state_df = getattr(env.state_builder, "state_df", None)
        if state_df is None or state_df.empty or "close" not in state_df:
            raise BacktestReplayError(
                "Backtest environment state is missing the 'close' column "
                "required for replay fills"
            )
        return float(state_df["close"].iloc[-1])

    def _transaction_cost_rate(
        self, request: TrainingDataRequest, include_costs: bool
    ) -> float:
        if not include_costs:
            return 0.0
        raw = getattr(request, "transaction_cost_rate", None)
        if raw is None:
            return _DEFAULT_TRANSACTION_COST_RATE
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            return _DEFAULT_TRANSACTION_COST_RATE


class ModelDrivenBacktestExecutor:
    """Default backtest executor that replays a trained generation.

    This executor implements the :class:`BacktestExecutor` protocol used by
    :class:`BacktestService`. It loads ``generation.model_path`` through
    :class:`StableBaselines3Trainer`, sources market/news data through
    :class:`DataAcquisitionService`, and replays the model over the sorted bars
    rather than running a model-independent momentum simulation.
    """

    def __init__(
        self,
        *,
        data_service: DataAcquisitionService,
        model_service: ModelService,
        project_root: Path | None = None,
        deterministic: bool = True,
    ) -> None:
        """Initialize the executor with the Phase 7 services it depends on.

        Args:
            data_service: Acquisition service for canonical market/news data.
            model_service: Model service for supporting-model validation.
            project_root: Optional override for the project data root.
            deterministic: Whether to use deterministic model predictions.
        """

        self._data_service = data_service
        self._model_service = model_service
        self._project_root = project_root
        self._engine = BacktestReplayEngine(
            data_service=data_service,
            model_service=model_service,
            project_root=project_root,
            deterministic=deterministic,
        )

    def execute(self, context: Any) -> Any:
        """Run a model-driven backtest and return trades plus equity snapshots.

        Args:
            context: A ``BacktestJobContext`` carrying the request, model, and
                generation.

        Returns:
            A ``BacktestExecutionResult`` with trades and an equity curve.

        Raises:
            BacktestReplayError: If the generation is not usable, required data
                is missing, or the model cannot be replayed.
        """

        # Imported lazily to avoid a circular import with the backtest service.
        from algotrading.api.services.backtest_service import BacktestExecutionResult

        request = context.request
        model = context.model
        generation = context.generation

        model_path = self._validate_generation(generation)
        symbols = _resolve_symbols(request, model)
        if not symbols:
            raise BacktestReplayError(
                "No symbols were supplied in the request or model "
                "training_data_config for the backtest"
            )

        allocation = float(request.initial_capital) / len(symbols)
        all_trades: list[TradeRecord] = []
        per_symbol_curves: list[list[EquityPoint]] = []

        for symbol in symbols:
            symbol_request = _build_symbol_request(request, model, symbol)
            self._ensure_data(symbol_request, model)
            replay = self._engine.run(
                model=model,
                model_path=model_path,
                request=symbol_request,
                initial_capital=allocation,
                include_costs=request.include_transaction_costs,
            )
            all_trades.extend(replay.trades)
            per_symbol_curves.append(replay.equity_curve)

        all_trades = _renumber_trades(all_trades)
        equity_curve = _aggregate_equity_curves(
            curves=per_symbol_curves,
            initial_capital=float(request.initial_capital),
        )
        if not equity_curve:
            raise BacktestReplayError("Backtest replay produced no equity curve")
        return BacktestExecutionResult(trades=all_trades, equity_curve=equity_curve)

    def _validate_generation(self, generation: Any) -> str:
        status = str(getattr(generation, "status", "")).lower()
        if status not in {"completed", "evaluated"}:
            raise BacktestReplayError(
                f"Generation '{getattr(generation, 'generation_id', '?')}' must be "
                f"completed before backtesting (current status: {status or 'unknown'})"
            )
        model_path = getattr(generation, "model_path", None)
        if not model_path:
            raise BacktestReplayError(
                f"Generation '{getattr(generation, 'generation_id', '?')}' has no "
                "model_path; the trained artifact is required for model-driven "
                "backtesting"
            )
        resolved = self._resolve_model_path(str(model_path))
        if resolved is None:
            raise BacktestReplayError(
                f"Trained model artifact for generation "
                f"'{getattr(generation, 'generation_id', '?')}' was not found at "
                f"'{model_path}'"
            )
        return resolved

    def _resolve_model_path(self, model_path: str) -> str | None:
        candidate = Path(model_path).expanduser()
        candidates = [candidate]
        if not candidate.is_absolute() and self._project_root is not None:
            candidates.append(Path(self._project_root) / candidate)
        for path in candidates:
            if path.exists():
                return str(path)
            zipped = Path(f"{path}.zip")
            if zipped.exists():
                return str(zipped)
        return None

    def _ensure_data(
        self, request: TrainingDataRequest, model: ModelConfigResponse
    ) -> None:
        try:
            self._data_service.ensure_market_data(request)
        except Exception as exc:
            raise BacktestReplayError(
                f"Required market data could not be sourced for backtest: {exc}"
            ) from exc

        if _requires_news(request, model, self._model_service):
            try:
                self._data_service.ensure_news_data(request, model=model)
            except Exception as exc:
                raise BacktestReplayError(
                    f"Required news data could not be sourced for backtest: {exc}"
                ) from exc


def build_backtest_environment(
    *,
    model: ModelConfigResponse,
    request: TrainingDataRequest,
    data_service: DataAcquisitionService,
    model_service: ModelService,
    project_root: Path | None = None,
) -> BuiltTradingEnvironment:
    """Build a single-pass ``TradingEnv`` for backtest replay.

    This reuses the Component 7.4 environment factory through the shared
    :func:`build_trading_environment` seam, requesting ``single_pass=True`` so
    the environment walks each sourced bar exactly once.

    Args:
        model: Core RL model whose config shapes the environment.
        request: Canonical data request (already merged for the backtest).
        data_service: Phase 7 acquisition service for canonical market/news.
        model_service: Model service for supporting-model validation.
        project_root: Optional override for the project data root.

    Returns:
        The built environment plus replay metadata.
    """

    return build_trading_environment(
        model=model,
        data_service=data_service,
        model_service=model_service,
        project_root=project_root,
        single_pass=True,
        request=request,
    )


def _build_symbol_request(
    request: Any, model: ModelConfigResponse, symbol: str
) -> TrainingDataRequest:
    """Build the canonical data request for one backtest symbol.

    By default the backtest widens to whole calendar days
    (``build_backtest_data_request``). However, ``BacktestRequest`` only carries
    dates, so when a model's ``training_data_config`` declares an explicit
    intraday window (for example IB 5-second bars from 15:00–15:08) that falls
    inside the requested dates, the backtest preserves that exact window instead
    of demanding a full day of bars the store does not have. This keeps the
    replay aligned with the data shape the model was trained on and lets cached
    intraday data satisfy the request.
    """

    intraday = _model_intraday_window(request, model)
    if intraday is not None:
        return normalize_training_data_request(
            model.training_data_config,
            {
                "symbols": [symbol],
                "start_time": intraday[0].isoformat(),
                "end_time": intraday[1].isoformat(),
            },
        )
    return build_backtest_data_request(
        request,
        model.training_data_config,
        overrides={"symbols": [symbol]},
    )


def _model_intraday_window(
    request: Any, model: ModelConfigResponse
) -> tuple[datetime, datetime] | None:
    """Return the model's intraday window when it fits the backtest dates."""

    config = model.training_data_config or {}
    raw_start = (
        config.get("start_time")
        or config.get("start_datetime")
        or config.get("from")
        or config.get("date_from")
    )
    raw_end = (
        config.get("end_time")
        or config.get("end_datetime")
        or config.get("to")
        or config.get("date_to")
    )
    if raw_start is None or raw_end is None:
        return None

    try:
        model_start = pd.to_datetime(raw_start, utc=True).to_pydatetime()
        model_end = pd.to_datetime(raw_end, utc=True).to_pydatetime()
    except (ValueError, TypeError):
        return None

    range_start = datetime.combine(request.start_date, datetime.min.time(), tzinfo=UTC)
    range_end = datetime.combine(request.end_date, datetime.max.time(), tzinfo=UTC)
    if model_start < range_start or model_end > range_end:
        return None
    if model_end <= model_start:
        return None
    return model_start, model_end


def _resolve_symbols(request: Any, model: ModelConfigResponse) -> list[str]:
    """Resolve requested symbols from the request or model config."""

    raw_symbols: object = getattr(request, "symbols", None)
    config = model.training_data_config
    if raw_symbols is None:
        raw_symbols = config.get("symbols")
    if raw_symbols is None:
        raw_symbols = config.get("symbol")

    if isinstance(raw_symbols, str):
        cleaned = raw_symbols.strip().upper()
        return [cleaned] if cleaned else []
    if isinstance(raw_symbols, (list, tuple)):
        return [str(item).strip().upper() for item in raw_symbols if str(item).strip()]
    return []


def _requires_news(
    request: TrainingDataRequest,
    model: ModelConfigResponse,
    model_service: ModelService,
) -> bool:
    """Return true when news must be sourced before replay."""

    if getattr(request.news_source, "enabled", False):
        return True
    try:
        from algotrading.src.data_pipeline.acquisition import model_requires_news

        return model_requires_news(model, model_service.supporting_registry)
    except Exception:
        # If we cannot determine the requirement, defer to explicit news config
        # rather than blocking a market-only backtest.
        return False


def _resolve_sb3_algorithm(model: ModelConfigResponse) -> SB3Algorithm:
    """Resolve the SB3 algorithm enum from the model's algorithm string."""

    algorithm_value = str(model.algorithm).lower()
    try:
        return SB3Algorithm(algorithm_value)
    except ValueError as exc:
        raise BacktestReplayError(
            f"Unsupported SB3 algorithm '{model.algorithm}' for backtest replay"
        ) from exc


def _resolve_policy(*, model: ModelConfigResponse, env: TradingEnv) -> str:
    """Resolve the SB3 policy name, mirroring the training factory default."""

    raw_policy = model.hyperparameters.get("model_policy") or model.hyperparameters.get(
        "policy"
    )
    if raw_policy is not None:
        return str(raw_policy)

    try:
        from gymnasium.spaces import Dict as DictSpace
    except Exception:
        return "MlpPolicy"

    if isinstance(getattr(env, "observation_space", None), DictSpace):
        return "MultiInputPolicy"
    return "MlpPolicy"


def _action_from_index(action_index: int) -> TradeAction:
    """Map a discrete model action index to a trade action label."""

    if action_index == _BUY_ACTION:
        return TradeAction.BUY
    if action_index == _SELL_ACTION:
        return TradeAction.SELL
    return TradeAction.HOLD


def _action_to_env_index(action: TradeAction) -> int:
    """Map a trade action label back to the environment action index."""

    if action == TradeAction.BUY:
        return _BUY_ACTION
    if action == TradeAction.SELL:
        return _SELL_ACTION
    return _HOLD_ACTION


def _renumber_trades(trades: list[TradeRecord]) -> list[TradeRecord]:
    """Assign stable sequential trade IDs in timestamp order across symbols."""

    ordered = sorted(trades, key=lambda trade: (trade.timestamp, trade.symbol))
    renumbered: list[TradeRecord] = []
    for index, trade in enumerate(ordered, start=1):
        renumbered.append(trade.model_copy(update={"trade_id": index}))
    return renumbered


def _aggregate_equity_curves(
    *, curves: list[list[EquityPoint]], initial_capital: float
) -> list[EquityPoint]:
    """Aggregate per-symbol equity curves into one portfolio curve."""

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
                timestamp=_to_utc_datetime(timestamp),
                portfolio_value=portfolio_value,
                cash=float(row["cash"]),
                position_value=float(row["position_value"]),
                drawdown=drawdown,
            )
        )
    return points


def _to_utc_datetime(value: Any) -> datetime:
    """Convert a date-like value into a timezone-aware UTC datetime."""

    if isinstance(value, datetime):
        parsed = value
    elif hasattr(value, "to_pydatetime"):
        parsed = value.to_pydatetime()
    else:
        parsed = pd.to_datetime(value, utc=True).to_pydatetime()

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


__all__ = [
    "BacktestReplayEngine",
    "BacktestReplayError",
    "ModelDrivenBacktestExecutor",
    "ReplayFillModel",
    "build_backtest_environment",
]
