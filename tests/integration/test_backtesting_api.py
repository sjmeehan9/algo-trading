"""Integration tests for Phase 5.4 backtesting API endpoints.

This module also covers Phase 7 Component 7.5 (model-driven backtesting
replay): the default ``BacktestService`` executor now loads the selected
trained generation and derives trades from model predictions over canonical
sourced market data instead of running a model-independent momentum
simulation.
"""

from __future__ import annotations

import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from algotrading.api.config import APIConfig
from algotrading.api.main import create_app
from algotrading.api.schemas.backtesting import (
    EquityPoint,
    TradeAction,
    TradeRecord,
)
from algotrading.api.schemas.data_sources import normalize_training_data_request
from algotrading.api.schemas.models import ModelConfigCreate, ModelType
from algotrading.api.schemas.training import TrainingJobStatus
from algotrading.api.services import ModelService
from algotrading.api.services import training_service as training_service_module
from algotrading.api.services.backtest_service import (
    BacktestExecutionResult,
    BacktestJobContext,
    BacktestService,
    create_default_backtest_service,
)
from algotrading.api.services.data_acquisition_service import (
    create_default_data_acquisition_service,
)
from algotrading.api.services.metrics_calculator import MetricsCalculator
from algotrading.src.backtesting.model_replay import (
    BacktestReplayEngine,
    ModelDrivenBacktestExecutor,
)
from algotrading.src.broker.registry import BrokerRegistry
from algotrading.src.data_pipeline.acquisition import HistoricalMarketDataAcquirer
from algotrading.src.data_pipeline.storage import LocalDataStore
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


# ---------------------------------------------------------------------------
# Phase 7 Component 7.5: Model-driven backtesting replay
# ---------------------------------------------------------------------------

_BACKTEST_API_KEY = "backtest-secret-key"


def _replay_market_csv(path: Path, rows: int = 48) -> tuple[datetime, datetime]:
    """Write deterministic minute bars used for model-driven replay tests.

    The close price oscillates with an upward drift so a trained policy has a
    non-degenerate price series to trade against.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    start = datetime(2024, 3, 4, 14, 30, tzinfo=UTC)
    records: list[dict[str, object]] = []
    for index in range(rows):
        timestamp = start + timedelta(minutes=index)
        open_price = 100.0 + (index * 0.1)
        close_price = open_price + (0.4 if index % 3 == 0 else -0.15)
        records.append(
            {
                "timestamp": timestamp.isoformat(),
                "symbol": "SPY",
                "open": open_price,
                "high": open_price + 0.5,
                "low": open_price - 0.5,
                "close": close_price,
                "volume": 1000 + index,
                "vwap": (open_price + close_price) / 2,
                "trade_count": 10 + index,
            }
        )
    pd.DataFrame(records).to_csv(path, index=False)
    return start, start + timedelta(minutes=rows - 1)


def _training_data_config(
    market_path: Path, start: datetime, end: datetime
) -> dict[str, object]:
    """Build a file-backed training data config for SPY minute bars."""

    return {
        "symbols": ["SPY"],
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "data_frequency": "1m",
        "market": {
            "provider": "file",
            "explicit_files": {"SPY": str(market_path)},
        },
    }


def _train_core_rl_model(
    *,
    app_state_model_service: ModelService,
    ws_manager: object,
    config: APIConfig,
    project_root: Path,
    market_path: Path,
    start: datetime,
    end: datetime,
    name: str = "Replay PPO",
    timesteps: int = 8,
) -> tuple[str, str, str]:
    """Train a tiny PPO model through the default service path.

    Returns the ``(model_id, generation_id, model_path)`` for the completed
    generation, with a real loadable SB3 artifact and canonical sourced data
    persisted under ``<project_root>/data/sourced``.
    """

    model = app_state_model_service.create_model(
        ModelConfigCreate(
            name=name,
            model_type=ModelType.CORE_RL,
            trainer_type="stable_baselines3",
            algorithm="ppo",
            hyperparameters={
                "learning_rate": 0.0003,
                "n_steps": 4,
                "batch_size": 4,
                "n_epochs": 1,
                "gamma": 0.95,
                "model_policy": "MultiInputPolicy",
            },
            training_data_config=_training_data_config(market_path, start, end),
            supporting_model_ids=[],
            strategy_ids=[],
            environment_config={"observation_window": 4},
            reward_function="profit_seeker",
        )
    )

    training_service = training_service_module.create_default_training_service(
        ws_manager=ws_manager,
        model_service=app_state_model_service,
        project_root=project_root,
        api_config=config,
    )

    deadline = time.time() + 25.0

    # Drive the training job through a TestClient so the worker starts and
    # drains the queue exactly like the production API path.
    app = create_app(config)
    app.state.model_service = app_state_model_service
    app.state.training_service = training_service
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/training/jobs",
            headers={"X-API-Key": config.api_key},
            json={"model_id": model.model_id, "total_timesteps": timesteps},
        )
        assert response.status_code == 201, response.text
        job_id = response.json()["data"]["job_id"]

        completed: dict[str, object] = {}
        while time.time() < deadline:
            poll = client.get(
                f"/api/v1/training/jobs/{job_id}",
                headers={"X-API-Key": config.api_key},
            )
            assert poll.status_code == 200, poll.text
            completed = poll.json()["data"]
            if completed["status"] == TrainingJobStatus.COMPLETED.value:
                break
            if completed["status"] == TrainingJobStatus.FAILED.value:
                raise AssertionError(f"training failed: {completed}")
            time.sleep(0.05)

        assert completed.get("status") == TrainingJobStatus.COMPLETED.value, completed
        generation_id = str(completed["generation_id"])
        generation = client.get(
            f"/api/v1/generations/{generation_id}",
            headers={"X-API-Key": config.api_key},
        ).json()["data"]
        model_path = str(generation["model_path"])

    assert Path(model_path).exists(), model_path
    return model.model_id, generation_id, model_path


def _model_driven_backtest_service(
    *,
    model_service: ModelService,
    config: APIConfig,
    project_root: Path,
    results_path: Path,
) -> BacktestService:
    """Build a default model-driven BacktestService bound to a temp store."""

    data_service = create_default_data_acquisition_service(
        api_config=config,
        project_root=project_root,
    )
    data_service._supporting_registry = model_service.supporting_registry
    service = create_default_backtest_service(
        model_service=model_service,
        project_root=project_root,
        data_service=data_service,
        api_config=config,
    )
    service._results_path = results_path
    return service


def test_default_model_driven_backtest_uses_trained_generation(
    tmp_path: Path,
) -> None:
    """Default backtest loads the trained generation and replays predictions."""

    config = APIConfig(
        api_key=_BACKTEST_API_KEY,
        debug=True,
        cors_origins=["http://localhost:3000"],
    )
    model_service = _build_model_service(tmp_path)
    market_path = tmp_path / "raw" / "spy.csv"
    start, end = _replay_market_csv(market_path)

    app = create_app(config)
    model_id, generation_id, model_path = _train_core_rl_model(
        app_state_model_service=model_service,
        ws_manager=app.state.ws_manager,
        config=config,
        project_root=tmp_path,
        market_path=market_path,
        start=start,
        end=end,
    )

    backtest_service = _model_driven_backtest_service(
        model_service=model_service,
        config=config,
        project_root=tmp_path,
        results_path=tmp_path / "backtest_results.json",
    )
    assert isinstance(backtest_service.executor, ModelDrivenBacktestExecutor)

    app.state.model_service = model_service
    app.state.backtest_service = backtest_service

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/backtests",
            headers={"X-API-Key": _BACKTEST_API_KEY},
            json={
                "model_id": model_id,
                "generation_id": generation_id,
                "start_date": start.date().isoformat(),
                "end_date": end.date().isoformat(),
                "initial_capital": 100_000.0,
                "symbols": ["SPY"],
            },
        )
        assert response.status_code == 201, response.text
        result = response.json()["data"]
        assert result["status"] == "completed", result
        assert result["equity_curve"], "equity curve must not be empty"
        # Equity points conform to the EquityPoint schema.
        for point in result["equity_curve"]:
            EquityPoint.model_validate(point)
        assert result["metrics"] is not None

        trades_response = client.get(
            f"/api/v1/backtests/{result['backtest_id']}/trades",
            headers={"X-API-Key": _BACKTEST_API_KEY},
        )
        assert trades_response.status_code == 200
        for trade in trades_response.json()["data"]:
            TradeRecord.model_validate(trade)

    # Canonical sourced store was used (Phase 7 data/sourced), not a CSV glob.
    sourced_market = tmp_path / "data" / "sourced" / "market" / "file" / "SPY"
    assert sourced_market.exists(), "backtest must read the canonical sourced store"
    assert list(sourced_market.rglob("market.csv")), "sourced market.csv missing"

    # Evaluation metrics were attached to the generation after the backtest.
    generation = model_service.generation_tracker.get_generation(generation_id)
    assert generation is not None
    assert generation.evaluation_metrics is not None


def test_model_driven_backtest_blocks_when_generation_artifact_missing(
    tmp_path: Path,
) -> None:
    """A generation whose artifact is missing fails the backtest clearly."""

    config = APIConfig(
        api_key=_BACKTEST_API_KEY,
        debug=True,
        cors_origins=["http://localhost:3000"],
    )
    model_service = _build_model_service(tmp_path)
    market_path = tmp_path / "raw" / "spy.csv"
    start, end = _replay_market_csv(market_path)

    model = model_service.create_model(
        ModelConfigCreate(
            name="No Artifact PPO",
            model_type=ModelType.CORE_RL,
            trainer_type="stable_baselines3",
            algorithm="ppo",
            hyperparameters={"model_policy": "MultiInputPolicy"},
            training_data_config=_training_data_config(market_path, start, end),
            supporting_model_ids=[],
            strategy_ids=[],
            environment_config={"observation_window": 4},
            reward_function="profit_seeker",
        )
    )
    generation = model_service.generation_tracker.start_generation(
        model_id=model.model_id,
        hyperparameters={},
    )
    model_service.generation_tracker.complete_generation(
        generation.generation_id,
        training_metrics=TrainingMetrics(
            final_reward=0.0,
            mean_reward=0.0,
            std_reward=0.0,
            episodes_completed=1,
            timesteps_trained=1,
            training_time_seconds=0.0,
        ),
        model_path=str(tmp_path / "models" / "does-not-exist.zip"),
    )

    backtest_service = _model_driven_backtest_service(
        model_service=model_service,
        config=config,
        project_root=tmp_path,
        results_path=tmp_path / "backtest_results.json",
    )
    app = create_app(config)
    app.state.model_service = model_service
    app.state.backtest_service = backtest_service

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/backtests",
            headers={"X-API-Key": _BACKTEST_API_KEY},
            json={
                "model_id": model.model_id,
                "generation_id": generation.generation_id,
                "start_date": start.date().isoformat(),
                "end_date": end.date().isoformat(),
                "initial_capital": 100_000.0,
                "symbols": ["SPY"],
            },
        )
        assert response.status_code == 201, response.text
        result = response.json()["data"]
        assert result["status"] == "failed"
        assert "model" in (result["error_message"] or "").lower()


class _ScriptedTrainer:
    """Deterministic trainer stub that emits a fixed action sequence."""

    def __init__(self, actions: list[int]) -> None:
        self._actions = actions
        self._index = 0
        self.is_trained = True

    def predict(
        self, observation: object, deterministic: bool = True
    ) -> tuple[int, dict[str, object]]:
        del observation, deterministic
        if self._index < len(self._actions):
            action = self._actions[self._index]
        else:
            action = 0
        self._index += 1
        return action, {}


def test_different_model_actions_produce_different_trades_on_same_data(
    tmp_path: Path,
) -> None:
    """Scope integrity: identical bars, different actions -> different trades.

    The replay engine walks a real file-backed environment once. Only the model
    action stream differs between the two runs, proving trades are derived from
    model predictions rather than deterministic price momentum.
    """

    config = APIConfig(
        api_key=_BACKTEST_API_KEY,
        debug=True,
        cors_origins=["http://localhost:3000"],
    )
    model_service = _build_model_service(tmp_path)
    market_path = tmp_path / "raw" / "spy.csv"
    start, end = _replay_market_csv(market_path)

    model = model_service.create_model(
        ModelConfigCreate(
            name="Action Stream PPO",
            model_type=ModelType.CORE_RL,
            trainer_type="stable_baselines3",
            algorithm="ppo",
            hyperparameters={"model_policy": "MultiInputPolicy"},
            training_data_config=_training_data_config(market_path, start, end),
            supporting_model_ids=[],
            strategy_ids=[],
            environment_config={"observation_window": 4},
            reward_function="profit_seeker",
        )
    )
    model_response = model_service.get_model(model.model_id)

    data_service = create_default_data_acquisition_service(
        api_config=config, project_root=tmp_path
    )
    data_service._supporting_registry = model_service.supporting_registry
    engine = BacktestReplayEngine(
        data_service=data_service,
        model_service=model_service,
        project_root=tmp_path,
    )

    request = normalize_training_data_request(
        _training_data_config(market_path, start, end), {}
    )
    data_service.ensure_market_data(request)

    # Action stream A: buy early, sell later. Stream B: never trade (all hold).
    buy_then_sell = [1, 0, 0, 0, 0, 2, 0, 0, 0, 0] + [0] * 64
    all_hold = [0] * 80

    def _run_with_actions(actions: list[int]) -> list[TradeRecord]:
        scripted = _ScriptedTrainer(list(actions))
        original_loader = engine._load_trainer
        engine._load_trainer = lambda **_: scripted  # type: ignore[assignment]
        try:
            replay = engine.run(
                model=model_response,
                model_path="unused-by-scripted-trainer",
                request=request,
                initial_capital=100_000.0,
                include_costs=True,
            )
        finally:
            engine._load_trainer = original_loader  # type: ignore[assignment]
        return replay.trades

    trades_a = _run_with_actions(buy_then_sell)
    trades_b = _run_with_actions(all_hold)

    actions_a = [(t.action.value, round(t.price, 4)) for t in trades_a]
    actions_b = [(t.action.value, round(t.price, 4)) for t in trades_b]
    assert actions_a != actions_b, "different actions must yield different trades"
    assert any(t.action == TradeAction.BUY for t in trades_a)
    # The all-hold stream never opens a position, so it never trades.
    assert trades_b == []


@pytest.mark.requires_ib
@pytest.mark.slow
def test_model_driven_backtest_after_ib_acquisition(
    confirm_ib_gateway: dict[str, object],
    tmp_path: Path,
) -> None:
    """Acquire live IB bars, then backtest the trained generation over them."""

    config = APIConfig(
        api_key=_BACKTEST_API_KEY,
        debug=True,
        cors_origins=["http://localhost:3000"],
    )
    model_service = _build_model_service(tmp_path)

    start = datetime(2026, 2, 13, 15, 0, tzinfo=UTC)
    end = start + timedelta(minutes=8)
    training_data_config = {
        "symbols": ["AMD"],
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "data_frequency": "5s",
        "cache_policy": "require_cache",
        "market": {
            "provider": "ib",
            "bar_size": "5 secs",
            "broker_data_type": "TRADES",
            "exchange": "SMART",
            "currency": "USD",
        },
    }

    store = LocalDataStore(root=tmp_path / "data" / "sourced")
    acquisition_request = normalize_training_data_request(
        training_data_config, {"cache_policy": "refresh"}
    )
    acquirer = HistoricalMarketDataAcquirer(
        store=store,
        broker_registry=BrokerRegistry.isolated(),
        broker_config={},
        broker_connection_params={
            "host": str(confirm_ib_gateway["host"]),
            "port": int(confirm_ib_gateway["port"]),
            "client_id": int(confirm_ib_gateway["client_id"]) + 81,
        },
    )
    acquisition_result = acquirer.acquire(acquisition_request)
    if not acquisition_result.reports or acquisition_result.reports[0].row_count == 0:
        pytest.fail("IB acquisition returned no bars for provider-backed backtest")

    model = model_service.create_model(
        ModelConfigCreate(
            name="IB Backed Replay PPO",
            model_type=ModelType.CORE_RL,
            trainer_type="stable_baselines3",
            algorithm="ppo",
            hyperparameters={
                "learning_rate": 0.0003,
                "n_steps": 4,
                "batch_size": 4,
                "n_epochs": 1,
                "model_policy": "MultiInputPolicy",
            },
            training_data_config=training_data_config,
            supporting_model_ids=[],
            strategy_ids=[],
            environment_config={"observation_window": 4},
            reward_function="profit_seeker",
        )
    )

    app = create_app(config)
    training_service = training_service_module.create_default_training_service(
        ws_manager=app.state.ws_manager,
        model_service=model_service,
        project_root=tmp_path,
        api_config=config,
    )
    app.state.model_service = model_service
    app.state.training_service = training_service

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/training/jobs",
            headers={"X-API-Key": _BACKTEST_API_KEY},
            json={"model_id": model.model_id, "total_timesteps": 8},
        )
        assert response.status_code == 201, response.text
        job_id = response.json()["data"]["job_id"]
        deadline = time.time() + 40.0
        completed: dict[str, object] = {}
        while time.time() < deadline:
            poll = client.get(
                f"/api/v1/training/jobs/{job_id}",
                headers={"X-API-Key": _BACKTEST_API_KEY},
            )
            completed = poll.json()["data"]
            if completed["status"] in {
                TrainingJobStatus.COMPLETED.value,
                TrainingJobStatus.FAILED.value,
            }:
                break
            time.sleep(0.1)
        assert completed.get("status") == TrainingJobStatus.COMPLETED.value, completed
        generation_id = str(completed["generation_id"])

    backtest_service = _model_driven_backtest_service(
        model_service=model_service,
        config=config,
        project_root=tmp_path,
        results_path=tmp_path / "backtest_results.json",
    )
    backtest_app = create_app(config)
    backtest_app.state.model_service = model_service
    backtest_app.state.backtest_service = backtest_service

    with TestClient(backtest_app) as client:
        response = client.post(
            "/api/v1/backtests",
            headers={"X-API-Key": _BACKTEST_API_KEY},
            json={
                "model_id": model.model_id,
                "generation_id": generation_id,
                "start_date": start.date().isoformat(),
                "end_date": end.date().isoformat(),
                "initial_capital": 50_000.0,
                "symbols": ["AMD"],
            },
        )
        assert response.status_code == 201, response.text
        result = response.json()["data"]
        assert result["status"] == "completed", result
        assert result["equity_curve"]

    sourced = tmp_path / "data" / "sourced" / "market" / "ib" / "AMD"
    assert sourced.exists()
