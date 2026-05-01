"""Unit tests for backtest service execution, storage, and comparison."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from algotrading.api.schemas.backtesting import (
    BacktestRequest,
    BacktestStatus,
    EquityPoint,
    TradeAction,
    TradeRecord,
)
from algotrading.api.schemas.models import ModelConfigCreate, ModelType
from algotrading.api.services import ModelService
from algotrading.api.services.backtest_service import (
    BacktestExecutionResult,
    BacktestJobContext,
    BacktestService,
    BacktestValidationError,
)
from algotrading.api.services.metrics_calculator import MetricsCalculator
from algotrading.src.models.registry import (
    CustomStrategyRegistry,
    SupportingModelRegistry,
)
from algotrading.src.models.tracking import (
    GenerationTracker,
    JsonFileStorage,
    TrainingMetrics,
)


class _DeterministicBacktestExecutor:
    """Backtest executor that returns deterministic trades and equity."""

    def __init__(self, final_value: float = 103_000.0) -> None:
        self.final_value = final_value
        self.contexts: list[BacktestJobContext] = []

    def execute(self, context: BacktestJobContext) -> BacktestExecutionResult:
        """Return a fixed profitable round-trip simulation."""

        self.contexts.append(context)
        start = datetime.combine(
            context.request.start_date, datetime.min.time(), tzinfo=UTC
        )
        end = start + timedelta(days=1)
        trades = [
            TradeRecord(
                trade_id=1,
                timestamp=start,
                symbol="AAPL",
                action=TradeAction.BUY,
                quantity=100.0,
                price=100.0,
                cost=0.0,
                position_after=100.0,
                portfolio_value=context.request.initial_capital,
            ),
            TradeRecord(
                trade_id=2,
                timestamp=end,
                symbol="AAPL",
                action=TradeAction.SELL,
                quantity=100.0,
                price=110.0,
                cost=0.0,
                position_after=0.0,
                portfolio_value=self.final_value,
            ),
        ]
        equity_curve = [
            EquityPoint(
                timestamp=start,
                portfolio_value=context.request.initial_capital,
                cash=context.request.initial_capital,
                position_value=0.0,
                drawdown=0.0,
            ),
            EquityPoint(
                timestamp=end,
                portfolio_value=self.final_value,
                cash=self.final_value,
                position_value=0.0,
                drawdown=0.0,
            ),
        ]
        return BacktestExecutionResult(trades=trades, equity_curve=equity_curve)


@pytest.fixture()
def model_service(tmp_path: Path) -> ModelService:
    """Create an isolated model service for backtest service tests."""

    return ModelService(
        supporting_registry=SupportingModelRegistry(),
        strategy_registry=CustomStrategyRegistry(strategy_dirs=[], auto_scan=False),
        generation_tracker=GenerationTracker(
            JsonFileStorage(str(tmp_path / "generations"))
        ),
        core_models_path=tmp_path / "core_models.json",
    )


def _create_model_and_generation(service: ModelService) -> tuple[str, str]:
    """Create a core model and completed generation for tests."""

    model = service.create_model(
        ModelConfigCreate(
            name="Backtest Core",
            model_type=ModelType.CORE_RL,
            trainer_type="stable_baselines3",
            algorithm="ppo",
            hyperparameters={"learning_rate": 0.0003, "n_steps": 2048},
            training_data_config={"symbols": ["AAPL"]},
            supporting_model_ids=[],
            strategy_ids=[],
            environment_config={},
            reward_function="profit_seeker",
        )
    )
    generation = service.generation_tracker.start_generation(
        model_id=model.model_id,
        hyperparameters={"learning_rate": 0.0003},
    )
    service.generation_tracker.complete_generation(
        generation.generation_id,
        training_metrics=TrainingMetrics(
            final_reward=1.0,
            mean_reward=0.8,
            std_reward=0.1,
            episodes_completed=3,
            timesteps_trained=100,
            training_time_seconds=1.5,
        ),
        model_path="models/backtest-core/model.zip",
    )
    return model.model_id, generation.generation_id


def _request(model_id: str, generation_id: str) -> BacktestRequest:
    """Build a baseline backtest request."""

    return BacktestRequest(
        model_id=model_id,
        generation_id=generation_id,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 2),
        initial_capital=100_000.0,
        symbols=["AAPL"],
    )


@pytest.mark.asyncio
async def test_run_backtest_persists_metrics_and_trades(
    tmp_path: Path,
    model_service: ModelService,
) -> None:
    """Backtest service should execute, store, and reload completed results."""

    model_id, generation_id = _create_model_and_generation(model_service)
    executor = _DeterministicBacktestExecutor()
    service = BacktestService(
        model_service=model_service,
        metrics_calculator=MetricsCalculator(),
        executor=executor,
        results_path=tmp_path / "backtests.json",
    )

    result = await service.run_backtest(_request(model_id, generation_id))

    assert result.status == BacktestStatus.COMPLETED
    assert result.metrics is not None
    assert result.metrics.total_return == pytest.approx(0.03)
    assert len(service.get_trades(result.backtest_id)) == 2
    assert executor.contexts[0].model.model_id == model_id

    reloaded = BacktestService(
        model_service=model_service,
        metrics_calculator=MetricsCalculator(),
        executor=_DeterministicBacktestExecutor(),
        results_path=tmp_path / "backtests.json",
    )
    assert reloaded.get_result(result.backtest_id).backtest_id == result.backtest_id
    assert len(reloaded.get_trades(result.backtest_id)) == 2


@pytest.mark.asyncio
async def test_compare_backtests_identifies_best_metrics(
    tmp_path: Path,
    model_service: ModelService,
) -> None:
    """Comparison should include metric arrays and best backtest IDs."""

    model_id, generation_id = _create_model_and_generation(model_service)
    service = BacktestService(
        model_service=model_service,
        metrics_calculator=MetricsCalculator(),
        executor=_DeterministicBacktestExecutor(final_value=101_000.0),
        results_path=tmp_path / "backtests.json",
    )
    first = await service.run_backtest(_request(model_id, generation_id))
    service.executor = _DeterministicBacktestExecutor(final_value=105_000.0)
    second = await service.run_backtest(_request(model_id, generation_id))

    comparison = service.compare_backtests([first.backtest_id, second.backtest_id])

    assert comparison.metric_comparison["total_return"] == pytest.approx([0.01, 0.05])
    assert comparison.best_by_metric["total_return"] == second.backtest_id


@pytest.mark.asyncio
async def test_backtest_requires_completed_generation(
    tmp_path: Path,
    model_service: ModelService,
) -> None:
    """Backtests should reject generations that are still training."""

    model = model_service.create_model(
        ModelConfigCreate(
            name="Unready Core",
            model_type=ModelType.CORE_RL,
            trainer_type="stable_baselines3",
            algorithm="ppo",
            hyperparameters={"learning_rate": 0.0003, "n_steps": 2048},
            training_data_config={"symbols": ["AAPL"]},
            supporting_model_ids=[],
            strategy_ids=[],
            environment_config={},
            reward_function="profit_seeker",
        )
    )
    generation = model_service.generation_tracker.start_generation(
        model_id=model.model_id,
        hyperparameters={"learning_rate": 0.0003},
    )
    service = BacktestService(
        model_service=model_service,
        metrics_calculator=MetricsCalculator(),
        executor=_DeterministicBacktestExecutor(),
        results_path=tmp_path / "backtests.json",
    )

    with pytest.raises(BacktestValidationError):
        await service.run_backtest(_request(model.model_id, generation.generation_id))
