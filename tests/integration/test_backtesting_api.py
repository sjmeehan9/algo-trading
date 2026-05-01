"""Integration tests for Phase 5.4 backtesting API endpoints."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from algotrading.api.config import APIConfig
from algotrading.api.main import create_app
from algotrading.api.schemas.backtesting import EquityPoint, TradeAction, TradeRecord
from algotrading.api.schemas.models import ModelConfigCreate, ModelType
from algotrading.api.services import ModelService
from algotrading.api.services.backtest_service import (
    BacktestExecutionResult,
    BacktestJobContext,
    BacktestService,
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
from fastapi.testclient import TestClient


class _ApiBacktestExecutor:
    """Deterministic executor used by API integration tests."""

    def execute(self, context: BacktestJobContext) -> BacktestExecutionResult:
        """Return fixed trade and equity output for endpoint tests."""

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
                portfolio_value=100_000.0,
            ),
            TradeRecord(
                trade_id=2,
                timestamp=end,
                symbol="AAPL",
                action=TradeAction.SELL,
                quantity=100.0,
                price=112.0,
                cost=0.0,
                position_after=0.0,
                portfolio_value=104_000.0,
            ),
        ]
        equity_curve = [
            EquityPoint(
                timestamp=start,
                portfolio_value=100_000.0,
                cash=100_000.0,
                position_value=0.0,
                drawdown=0.0,
            ),
            EquityPoint(
                timestamp=end,
                portfolio_value=104_000.0,
                cash=104_000.0,
                position_value=0.0,
                drawdown=0.0,
            ),
        ]
        return BacktestExecutionResult(trades=trades, equity_curve=equity_curve)


def _build_model_service(tmp_path: Path) -> ModelService:
    """Build an isolated model service for API tests."""

    return ModelService(
        supporting_registry=SupportingModelRegistry(),
        strategy_registry=CustomStrategyRegistry(strategy_dirs=[], auto_scan=False),
        generation_tracker=GenerationTracker(
            JsonFileStorage(str(tmp_path / "generations"))
        ),
        core_models_path=tmp_path / "core_models.json",
    )


def _create_model_and_generation(service: ModelService) -> tuple[str, str]:
    """Create a model and completed generation via service APIs."""

    model = service.create_model(
        ModelConfigCreate(
            name="Backtest API Core",
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
            final_reward=1.5,
            mean_reward=1.2,
            std_reward=0.1,
            episodes_completed=4,
            timesteps_trained=200,
            training_time_seconds=2.0,
        ),
        model_path="models/api-backtest/model.zip",
    )
    return model.model_id, generation.generation_id


@pytest.fixture()
def api_client(tmp_path: Path) -> tuple[TestClient, ModelService]:
    """Create a TestClient with isolated model and backtest services."""

    config = APIConfig(
        api_key="backtest-secret-key",
        debug=True,
        cors_origins=["http://localhost:3000"],
    )
    app = create_app(config)
    model_service = _build_model_service(tmp_path)
    app.state.model_service = model_service
    app.state.backtest_service = BacktestService(
        model_service=model_service,
        metrics_calculator=MetricsCalculator(),
        executor=_ApiBacktestExecutor(),
        results_path=tmp_path / "backtest_results.json",
    )

    with TestClient(app) as client:
        yield client, model_service


def _auth_headers() -> dict[str, str]:
    """Return API authentication headers."""

    return {"X-API-Key": "backtest-secret-key"}


def _payload(model_id: str, generation_id: str) -> dict[str, object]:
    """Build a valid backtest request payload."""

    return {
        "model_id": model_id,
        "generation_id": generation_id,
        "start_date": date(2026, 1, 1).isoformat(),
        "end_date": date(2026, 1, 2).isoformat(),
        "initial_capital": 100_000.0,
        "symbols": ["AAPL"],
    }


def test_backtest_api_run_retrieve_trades_and_compare(
    api_client: tuple[TestClient, ModelService],
) -> None:
    """Endpoint flow should run, retrieve, list, trade-read, and compare."""

    client, model_service = api_client
    model_id, generation_id = _create_model_and_generation(model_service)

    first_response = client.post(
        "/api/v1/backtests",
        headers=_auth_headers(),
        json=_payload(model_id, generation_id),
    )
    assert first_response.status_code == 201, first_response.text
    first = first_response.json()["data"]
    assert first["status"] == "completed"
    assert first["metrics"]["total_return"] == pytest.approx(0.04)

    second_response = client.post(
        "/api/v1/backtests",
        headers=_auth_headers(),
        json=_payload(model_id, generation_id),
    )
    assert second_response.status_code == 201, second_response.text
    second = second_response.json()["data"]

    detail_response = client.get(
        f"/api/v1/backtests/{first['backtest_id']}", headers=_auth_headers()
    )
    assert detail_response.status_code == 200
    assert detail_response.json()["data"]["backtest_id"] == first["backtest_id"]

    trades_response = client.get(
        f"/api/v1/backtests/{first['backtest_id']}/trades",
        headers=_auth_headers(),
    )
    assert trades_response.status_code == 200
    assert len(trades_response.json()["data"]) == 2

    list_response = client.get(
        "/api/v1/backtests",
        headers=_auth_headers(),
        params={"model_id": model_id},
    )
    assert list_response.status_code == 200
    assert list_response.json()["total"] == 2

    compare_response = client.post(
        "/api/v1/backtests/compare",
        headers=_auth_headers(),
        json={"backtest_ids": [first["backtest_id"], second["backtest_id"]]},
    )
    assert compare_response.status_code == 200
    comparison = compare_response.json()["data"]
    assert comparison["best_by_metric"]["total_return"] in {
        first["backtest_id"],
        second["backtest_id"],
    }


def test_backtest_api_missing_generation_returns_404(
    api_client: tuple[TestClient, ModelService],
) -> None:
    """Unknown generation IDs should return standardized 404 errors."""

    client, model_service = api_client
    model_id, _generation_id = _create_model_and_generation(model_service)

    response = client.post(
        "/api/v1/backtests",
        headers=_auth_headers(),
        json=_payload(model_id, "missing-generation"),
    )

    assert response.status_code == 404
    assert response.json()["error_code"] == "GENERATION_NOT_FOUND"
