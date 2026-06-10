"""Phase 7.6 file-backed regression for the data-backed backtesting workflow.

This suite proves the backtest workflow is no longer decoupled from the trained
generation. A tiny PPO model is trained through the default service path so a
real Stable Baselines 3 artifact and canonical sourced market data exist, then
the default model-driven backtest loads that artifact and replays the model's
own predictions over the canonical store, producing trades, an equity curve,
and evaluation metrics attached back to the generation.

The tests are deliberately tiny (small window, low timesteps) so they run
routinely in CI with no provider flags. Provider-gated acquisition coverage is
in ``test_data_acquisition_market.py`` and ``test_data_acquisition_news.py``.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from algotrading.api.config import APIConfig
from algotrading.api.main import create_app
from algotrading.api.schemas.backtesting import EquityPoint, TradeRecord
from algotrading.api.schemas.models import ModelConfigCreate, ModelType
from algotrading.api.schemas.training import TrainingJobStatus
from algotrading.api.services import ModelService
from algotrading.api.services import training_service as training_service_module
from algotrading.api.services.backtest_service import (
    create_default_backtest_service,
)
from algotrading.api.services.data_acquisition_service import (
    create_default_data_acquisition_service,
)
from algotrading.src.backtesting.model_replay import ModelDrivenBacktestExecutor
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

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

_API_KEY = "backtest-workflow-secret-key"


def _auth_headers() -> dict[str, str]:
    return {"X-API-Key": _API_KEY}


def _build_model_service(tmp_path: Path) -> ModelService:
    """Build an isolated model service writing to a temp directory."""

    return ModelService(
        supporting_registry=SupportingModelRegistry(),
        strategy_registry=CustomStrategyRegistry(strategy_dirs=[], auto_scan=False),
        generation_tracker=GenerationTracker(
            JsonFileStorage(str(tmp_path / "generations"))
        ),
        core_models_path=tmp_path / "core_models.json",
    )


def _write_market_csv(path: Path, rows: int = 48) -> tuple[datetime, datetime]:
    """Write deterministic minute bars with drift for a non-degenerate series."""

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


def _create_core_model(
    service: ModelService,
    training_data_config: dict[str, object],
    *,
    name: str = "Backtest Workflow PPO",
) -> Any:
    """Persist a tiny core RL model and return the model response."""

    return service.create_model(
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
            training_data_config=training_data_config,
            supporting_model_ids=[],
            strategy_ids=[],
            environment_config={"observation_window": 4},
            reward_function="profit_seeker",
        )
    )


def _train_core_rl_model(
    *,
    model_service: ModelService,
    config: APIConfig,
    project_root: Path,
    model_id: str,
    timesteps: int = 8,
    timeout: float = 25.0,
) -> tuple[str, str]:
    """Train a model through the default service path via the API.

    Returns ``(generation_id, model_path)`` for the completed generation, with a
    real loadable SB3 artifact and canonical sourced data persisted under
    ``<project_root>/data/sourced``.
    """

    app = create_app(config)
    app.state.model_service = model_service
    app.state.training_service = (
        training_service_module.create_default_training_service(
            ws_manager=app.state.ws_manager,
            model_service=model_service,
            project_root=project_root,
            api_config=config,
        )
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/training/jobs",
            headers=_auth_headers(),
            json={"model_id": model_id, "total_timesteps": timesteps},
        )
        assert response.status_code == 201, response.text
        job_id = response.json()["data"]["job_id"]

        deadline = time.time() + timeout
        completed: dict[str, Any] = {}
        while time.time() < deadline:
            poll = client.get(
                f"/api/v1/training/jobs/{job_id}",
                headers=_auth_headers(),
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
            headers=_auth_headers(),
        ).json()["data"]
        model_path = str(generation["model_path"])

    assert Path(model_path).exists(), model_path
    return generation_id, model_path


def _model_driven_backtest_app(
    *,
    model_service: ModelService,
    config: APIConfig,
    project_root: Path,
    results_path: Path,
) -> Any:
    """Build an app with a default model-driven BacktestService attached."""

    data_service = create_default_data_acquisition_service(
        api_config=config,
        project_root=project_root,
    )
    data_service._supporting_registry = model_service.supporting_registry
    backtest_service = create_default_backtest_service(
        model_service=model_service,
        project_root=project_root,
        data_service=data_service,
        api_config=config,
    )
    backtest_service._results_path = results_path
    assert isinstance(backtest_service.executor, ModelDrivenBacktestExecutor)

    app = create_app(config)
    app.state.model_service = model_service
    app.state.backtest_service = backtest_service
    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_backtest_loads_trained_artifact_and_replays_over_canonical_store(
    tmp_path: Path,
) -> None:
    """End-to-end: train -> generation artifact -> backtest -> metrics.

    Asserts observable behavior rather than only HTTP 201:

    * The backtest completes and produces an equity curve and metrics.
    * The trained generation's model artifact exists and is the one replayed.
    * The canonical sourced store (data/sourced) is read, not a CSV glob.
    * Evaluation metrics are attached back to the generation.
    """

    config = APIConfig(
        api_key=_API_KEY,
        debug=True,
        cors_origins=["http://localhost:3000"],
    )
    model_service = _build_model_service(tmp_path)

    market_path = tmp_path / "raw" / "spy.csv"
    start, end = _write_market_csv(market_path)
    model = _create_core_model(
        model_service, _training_data_config(market_path, start, end)
    )

    generation_id, model_path = _train_core_rl_model(
        model_service=model_service,
        config=config,
        project_root=tmp_path,
        model_id=model.model_id,
    )

    app = _model_driven_backtest_app(
        model_service=model_service,
        config=config,
        project_root=tmp_path,
        results_path=tmp_path / "backtest_results.json",
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/backtests",
            headers=_auth_headers(),
            json={
                "model_id": model.model_id,
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
        for point in result["equity_curve"]:
            EquityPoint.model_validate(point)
        assert result["metrics"] is not None

        trades_response = client.get(
            f"/api/v1/backtests/{result['backtest_id']}/trades",
            headers=_auth_headers(),
        )
        assert trades_response.status_code == 200
        for trade in trades_response.json()["data"]:
            TradeRecord.model_validate(trade)

    # Observable behavior 1: the artifact replayed is the trained generation's.
    assert Path(model_path).exists()

    # Observable behavior 2: the canonical sourced store was used (Phase 7), not
    # a CSV glob over saved_data.
    sourced_market = tmp_path / "data" / "sourced" / "market" / "file" / "SPY"
    assert sourced_market.exists(), "backtest must read the canonical sourced store"
    assert list(sourced_market.rglob("market.csv")), "sourced market.csv missing"

    # Observable behavior 3: evaluation metrics were attached to the generation.
    generation = model_service.generation_tracker.get_generation(generation_id)
    assert generation is not None
    assert generation.evaluation_metrics is not None


def test_backtest_fails_for_the_right_reason_when_artifact_missing(
    tmp_path: Path,
) -> None:
    """A generation whose artifact file is missing fails the backtest cleanly.

    The model-driven executor must refuse to fabricate a model-independent
    simulation when the artifact is absent; it records a failed backtest with a
    model/artifact-specific message rather than completing.
    """

    config = APIConfig(
        api_key=_API_KEY,
        debug=True,
        cors_origins=["http://localhost:3000"],
    )
    model_service = _build_model_service(tmp_path)

    market_path = tmp_path / "raw" / "spy.csv"
    start, end = _write_market_csv(market_path)
    model = _create_core_model(
        model_service,
        _training_data_config(market_path, start, end),
        name="Missing Artifact PPO",
    )

    # Create a completed generation that points at a non-existent artifact.
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

    app = _model_driven_backtest_app(
        model_service=model_service,
        config=config,
        project_root=tmp_path,
        results_path=tmp_path / "backtest_results.json",
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/backtests",
            headers=_auth_headers(),
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
        assert result["status"] == "failed", result
        assert "model" in (result["error_message"] or "").lower()
