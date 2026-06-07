"""Integration tests for Phase 5.3 training control API endpoints."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from algotrading.api.config import APIConfig
from algotrading.api.main import create_app
from algotrading.api.schemas.data_sources import normalize_training_data_request
from algotrading.api.schemas.models import ModelConfigCreate, ModelType
from algotrading.api.schemas.training import TrainingJobStatus
from algotrading.api.services import ModelService
from algotrading.api.services import training_service as training_service_module
from algotrading.api.services.data_acquisition_service import (
    create_default_data_acquisition_service,
)
from algotrading.api.services.training_service import TrainingService
from algotrading.api.training.factories import (
    TrainingFactoryError,
    build_core_rl_environment,
    build_news_sentiment_dataset,
)
from algotrading.api.workers.training_worker import (
    TrainingCancelledError,
    TrainingExecutionResult,
    TrainingJobContext,
    TrainingProgressUpdate,
    TrainingWorker,
)
from algotrading.src.broker.registry import BrokerRegistry
from algotrading.src.data_pipeline.acquisition import HistoricalMarketDataAcquirer
from algotrading.src.data_pipeline.storage import LocalDataStore
from algotrading.src.data_pipeline.types import NewsRecord
from algotrading.src.models.registry import (
    CustomStrategyRegistry,
    SupportingModelRegistry,
)
from algotrading.src.models.signals import SignalType
from algotrading.src.models.supporting.sentiment.news_sentiment import (
    NewsSentimentTrainer,
)
from algotrading.src.models.tracking import (
    GenerationTracker,
    JsonFileStorage,
    TrainingMetrics,
)
from algotrading.src.trainers.sb3_trainer import SB3Algorithm, StableBaselines3Trainer
from fastapi.testclient import TestClient


class _DeterministicExecutor:
    """Executor that emits deterministic progress and completes successfully."""

    def __init__(self) -> None:
        self.invocations: list[TrainingJobContext] = []

    def execute(self, context: TrainingJobContext) -> TrainingExecutionResult:
        self.invocations.append(context)
        total = max(1, context.job.total_timesteps)
        for step in (1, 2):
            context.progress_callback(
                TrainingProgressUpdate(
                    current_timestep=int(total * (step / 2)),
                    total_timesteps=total,
                    current_metrics={"reward": float(step)},
                )
            )
        training_metrics = TrainingMetrics(
            final_reward=2.0,
            mean_reward=1.5,
            std_reward=0.5,
            episodes_completed=2,
            timesteps_trained=total,
            training_time_seconds=0.01,
        )
        return TrainingExecutionResult(
            timesteps_trained=total,
            final_metrics={"final_reward": 2.0, "episodes_completed": 2},
            training_metrics=training_metrics,
            model_path="/tmp/model.zip",
        )


class _BlockingExecutor:
    """Executor that stays running until cancellation is requested."""

    def execute(self, context: TrainingJobContext) -> TrainingExecutionResult:
        context.progress_callback(
            TrainingProgressUpdate(
                current_timestep=1,
                total_timesteps=max(1, context.job.total_timesteps),
                current_metrics={"phase": "running"},
            )
        )
        deadline = time.time() + 10.0
        while not context.cancellation_event.is_set() and time.time() < deadline:
            time.sleep(0.01)
        if context.cancellation_event.is_set():
            raise TrainingCancelledError("cancelled")
        raise RuntimeError("blocking executor timed out waiting for cancellation")


def _build_model_service(tmp_path: Path) -> ModelService:
    """Build an isolated model service for API tests."""

    supporting_registry = SupportingModelRegistry()
    strategy_registry = CustomStrategyRegistry(strategy_dirs=[], auto_scan=False)
    generation_tracker = GenerationTracker(
        JsonFileStorage(str(tmp_path / "generations"))
    )
    return ModelService(
        supporting_registry=supporting_registry,
        strategy_registry=strategy_registry,
        generation_tracker=generation_tracker,
        core_models_path=tmp_path / "core_models.json",
    )


def _create_core_model(service: ModelService) -> str:
    """Persist a core RL model and return its ID."""

    response = service.create_model(
        ModelConfigCreate(
            name="API Core",
            model_type=ModelType.CORE_RL,
            trainer_type="stable_baselines3",
            algorithm="ppo",
            hyperparameters={"learning_rate": 0.0003, "n_steps": 2048},
            training_data_config={},
            supporting_model_ids=[],
            strategy_ids=[],
            environment_config={},
            reward_function="profit_seeker",
        )
    )
    return response.model_id


def _create_supporting_ml_model(
    service: ModelService,
    training_data_config: dict[str, object] | None = None,
) -> str:
    """Persist a supporting sentiment model and return its ID."""

    response = service.create_model(
        ModelConfigCreate(
            name="API Sentiment",
            model_type=ModelType.SUPPORTING_ML,
            signal_type=SignalType.SENTIMENT,
            trainer_type="sklearn",
            algorithm="random_forest",
            hyperparameters={"epochs": 1},
            training_data_config=training_data_config or {},
            input_data_types=["news_text"],
            input_frequency="irregular",
        )
    )
    return response.model_id


def _write_market_csv(path: Path, rows: int = 24) -> tuple[datetime, datetime]:
    """Write deterministic minute bars for file-backed training tests."""

    path.parent.mkdir(parents=True, exist_ok=True)
    start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    records: list[dict[str, object]] = []
    for index in range(rows):
        timestamp = start + timedelta(minutes=index)
        open_price = 100.0 + (index * 0.1)
        close_price = open_price + (0.05 if index % 2 == 0 else -0.02)
        records.append(
            {
                "timestamp": timestamp.isoformat(),
                "symbol": "SPY",
                "open": open_price,
                "high": open_price + 0.2,
                "low": open_price - 0.2,
                "close": close_price,
                "volume": 1000 + index,
                "vwap": (open_price + close_price) / 2,
                "trade_count": 10 + index,
            }
        )
    pd.DataFrame(records).to_csv(path, index=False)
    return start, start + timedelta(minutes=rows - 1)


def _write_sentiment_csv(path: Path) -> Path:
    """Write a tiny labeled sentiment dataset for supporting ML tests."""

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "headline": "Strong earnings beat expectations",
                "body": "Guidance improved after resilient demand.",
                "sentiment_score": 0.75,
            },
            {
                "headline": "Margins narrow after weaker sales",
                "body": "Management warned on near-term pressure.",
                "sentiment_score": -0.55,
            },
            {
                "headline": "Analysts see stable outlook",
                "body": "The company maintained its full-year forecast.",
                "sentiment_score": 0.15,
            },
        ]
    ).to_csv(path, index=False)
    return path


@pytest.fixture()
def api_client(
    tmp_path: Path,
) -> tuple[TestClient, ModelService, _DeterministicExecutor]:
    """Build a TestClient with model + training services wired in."""

    config = APIConfig(
        api_key="training-secret-key",
        debug=True,
        cors_origins=["http://localhost:3000"],
    )
    app = create_app(config)

    model_service = _build_model_service(tmp_path)
    app.state.model_service = model_service

    executor = _DeterministicExecutor()
    worker = TrainingWorker(executor=executor)
    training_service = TrainingService(
        ws_manager=app.state.ws_manager,
        model_service=model_service,
        generation_tracker=model_service.generation_tracker,
        worker=worker,
        jobs_path=tmp_path / "training_jobs.json",
    )
    app.state.training_service = training_service

    with TestClient(app) as client:
        yield client, model_service, executor


def _auth_headers() -> dict[str, str]:
    """Return API key headers for protected endpoints."""

    return {"X-API-Key": "training-secret-key"}


def _wait_for_status(
    client: TestClient,
    job_id: str,
    status: str,
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Poll the get-job endpoint until the job reaches the desired status."""

    deadline = time.time() + timeout
    last: dict[str, Any] = {}
    while time.time() < deadline:
        response = client.get(
            f"/api/v1/training/jobs/{job_id}", headers=_auth_headers()
        )
        assert response.status_code == 200, response.text
        last = response.json()["data"]
        if last["status"] == status:
            return last
        time.sleep(0.05)
    raise AssertionError(
        f"Job {job_id} did not reach status {status}; last={last.get('status')}"
    )


def test_create_job_for_unknown_model_returns_400(
    api_client: tuple[TestClient, ModelService, _DeterministicExecutor],
) -> None:
    """Creating a job referencing a missing model returns a validation error."""

    client, _service, _executor = api_client
    response = client.post(
        "/api/v1/training/jobs",
        headers=_auth_headers(),
        json={"model_id": "no-such-model"},
    )
    assert response.status_code == 400
    assert response.json()["error_code"] == "TRAINING_VALIDATION_ERROR"


def test_full_training_job_lifecycle(
    api_client: tuple[TestClient, ModelService, _DeterministicExecutor],
) -> None:
    """Create -> queue -> run -> complete via API and verify generation linkage."""

    client, model_service, executor = api_client
    model_id = _create_core_model(model_service)

    response = client.post(
        "/api/v1/training/jobs",
        headers=_auth_headers(),
        json={"model_id": model_id, "total_timesteps": 50},
    )
    assert response.status_code == 201
    created = response.json()["data"]
    job_id = created["job_id"]
    assert created["status"] == TrainingJobStatus.QUEUED.value

    completed = _wait_for_status(client, job_id, TrainingJobStatus.COMPLETED.value)

    assert completed["progress_percent"] == 100.0
    assert completed["current_metrics"]["final_reward"] == 2.0
    assert completed["generation_id"] is not None
    assert executor.invocations, "executor should be invoked"

    list_response = client.get(
        "/api/v1/training/jobs",
        headers=_auth_headers(),
        params={"model_id": model_id},
    )
    assert list_response.status_code == 200
    listed = list_response.json()["data"]
    assert any(item["job_id"] == job_id for item in listed)

    # Generation history exposes the new generation via the model endpoint.
    generations_response = client.get(
        f"/api/v1/models/{model_id}/generations", headers=_auth_headers()
    )
    assert generations_response.status_code == 200
    items = generations_response.json()["items"]
    assert any(item["generation_id"] == completed["generation_id"] for item in items)


def test_lazy_training_service_starts_worker_and_drains_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lazily created training service starts its worker when a job is queued."""

    config = APIConfig(
        api_key="training-secret-key",
        debug=True,
        cors_origins=["http://localhost:3000"],
    )
    app = create_app(config)

    model_service = _build_model_service(tmp_path)
    app.state.model_service = model_service

    executor = _DeterministicExecutor()
    created_services: list[TrainingService] = []

    def _factory(
        *,
        ws_manager: Any,
        model_service: ModelService,
        project_root: Path | None = None,
        api_config: APIConfig | None = None,
        worker: TrainingWorker | None = None,
    ) -> TrainingService:
        del project_root, api_config, worker
        lazy_worker = TrainingWorker(executor=executor)
        service = TrainingService(
            ws_manager=ws_manager,
            model_service=model_service,
            generation_tracker=model_service.generation_tracker,
            worker=lazy_worker,
            jobs_path=tmp_path / "lazy_training_jobs.json",
        )
        created_services.append(service)
        return service

    monkeypatch.setattr(
        training_service_module,
        "create_default_training_service",
        _factory,
    )

    model_id = _create_core_model(model_service)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/training/jobs",
            headers=_auth_headers(),
            json={"model_id": model_id, "total_timesteps": 20},
        )

        assert response.status_code == 201, response.text
        created = response.json()["data"]
        job_id = created["job_id"]
        assert created["status"] == TrainingJobStatus.QUEUED.value
        assert created_services
        assert created_services[0].worker.is_running

        completed = _wait_for_status(
            client,
            job_id,
            TrainingJobStatus.COMPLETED.value,
        )

        assert completed["started_at"] is not None
        assert completed["generation_id"] is not None
        assert executor.invocations


def test_preattached_training_service_starts_and_stops_on_lifespan(
    tmp_path: Path,
) -> None:
    """A pre-attached training service still starts on startup and stops on shutdown."""

    config = APIConfig(
        api_key="training-secret-key",
        debug=True,
        cors_origins=["http://localhost:3000"],
    )
    app = create_app(config)

    model_service = _build_model_service(tmp_path)
    app.state.model_service = model_service

    worker = TrainingWorker(executor=_DeterministicExecutor())
    training_service = TrainingService(
        ws_manager=app.state.ws_manager,
        model_service=model_service,
        generation_tracker=model_service.generation_tracker,
        worker=worker,
        jobs_path=tmp_path / "pre_attached_training_jobs.json",
    )
    app.state.training_service = training_service

    assert not worker.is_running

    with TestClient(app) as client:
        health_response = client.get("/health")
        assert health_response.status_code == 200
        assert worker.is_running

    assert not worker.is_running


def test_default_service_trains_core_rl_with_file_backed_market_data(
    tmp_path: Path,
) -> None:
    """Default API service builds a real TradingEnv and saves an SB3 artifact."""

    config = APIConfig(
        api_key="training-secret-key",
        debug=True,
        cors_origins=["http://localhost:3000"],
    )
    app = create_app(config)

    model_service = _build_model_service(tmp_path)
    app.state.model_service = model_service

    market_path = tmp_path / "raw" / "spy.csv"
    start, end = _write_market_csv(market_path)
    model = model_service.create_model(
        ModelConfigCreate(
            name="File Backed PPO",
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
            training_data_config={
                "symbols": ["SPY"],
                "start_time": start.isoformat(),
                "end_time": end.isoformat(),
                "data_frequency": "1m",
                "market": {
                    "provider": "file",
                    "explicit_files": {"SPY": str(market_path)},
                },
            },
            supporting_model_ids=[],
            strategy_ids=[],
            environment_config={"observation_window": 4},
            reward_function="profit_seeker",
        )
    )

    app.state.training_service = (
        training_service_module.create_default_training_service(
            ws_manager=app.state.ws_manager,
            model_service=model_service,
            project_root=tmp_path,
            api_config=config,
        )
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/training/jobs",
            headers=_auth_headers(),
            json={"model_id": model.model_id, "total_timesteps": 8},
        )
        assert response.status_code == 201, response.text
        created = response.json()["data"]
        job_id = created["job_id"]

        assert created["status"] == TrainingJobStatus.QUEUED.value
        assert created["started_at"] is None
        assert created["generation_id"] is None
        assert app.state.training_service.worker.is_running

        completed = _wait_for_status(
            client,
            job_id,
            TrainingJobStatus.COMPLETED.value,
            timeout=20.0,
        )

        assert completed["started_at"] is not None
        assert completed["completed_at"] is not None
        assert completed["generation_id"] is not None
        assert completed["progress_percent"] == 100.0
        assert completed["current_timestep"] == 8

        generation_response = client.get(
            f"/api/v1/generations/{completed['generation_id']}",
            headers=_auth_headers(),
        )
        assert generation_response.status_code == 200, generation_response.text
        generation = generation_response.json()["data"]
        assert generation["generation_id"] == completed["generation_id"]
        assert generation["model_id"] == model.model_id
        assert generation["training_config"]["status"] == "completed"
        assert generation["training_duration_seconds"] >= 0.0
        assert generation["metrics"]["timesteps_trained"] == 8

        model_path = Path(generation["model_path"])
        assert model_path.exists()

        generations_response = client.get(
            f"/api/v1/models/{model.model_id}/generations",
            headers=_auth_headers(),
        )
        assert generations_response.status_code == 200, generations_response.text
        assert any(
            item["generation_id"] == completed["generation_id"]
            and item["final_reward"] is not None
            for item in generations_response.json()["items"]
        )

        reload_data_service = create_default_data_acquisition_service(
            api_config=config,
            project_root=tmp_path,
        )
        load_env = build_core_rl_environment(
            model=model,
            data_config={"total_timesteps": 8},
            data_service=reload_data_service,
            model_service=model_service,
            project_root=tmp_path,
        )

        trainer = StableBaselines3Trainer(
            algorithm=SB3Algorithm.PPO,
            policy="MultiInputPolicy",
        )
        trainer.load(str(model_path), env=load_env)
        assert trainer.is_trained


def test_default_service_trains_supporting_ml_with_labeled_dataset(
    tmp_path: Path,
) -> None:
    """Default API service trains supporting ML using the dataset factory."""

    config = APIConfig(
        api_key="training-secret-key",
        debug=True,
        cors_origins=["http://localhost:3000"],
    )
    app = create_app(config)

    model_service = _build_model_service(tmp_path)
    app.state.model_service = model_service
    dataset_path = _write_sentiment_csv(tmp_path / "raw" / "sentiment.csv")
    model_id = _create_supporting_ml_model(model_service)

    app.state.training_service = (
        training_service_module.create_default_training_service(
            ws_manager=app.state.ws_manager,
            model_service=model_service,
            project_root=tmp_path,
            api_config=config,
        )
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/training/jobs",
            headers=_auth_headers(),
            json={
                "model_id": model_id,
                "total_timesteps": 1,
                "data_config": {"labeled_dataset_path": str(dataset_path)},
            },
        )
        assert response.status_code == 201, response.text
        job_id = response.json()["data"]["job_id"]

        completed = _wait_for_status(
            client,
            job_id,
            TrainingJobStatus.COMPLETED.value,
            timeout=10.0,
        )

        assert completed["started_at"] is not None
        assert completed["completed_at"] is not None
        assert completed["generation_id"] is not None
        assert completed["progress_percent"] == 100.0
        assert completed["current_metrics"]["epochs_trained"] == 1

        generation_response = client.get(
            f"/api/v1/generations/{completed['generation_id']}",
            headers=_auth_headers(),
        )
        assert generation_response.status_code == 200, generation_response.text
        generation = generation_response.json()["data"]
        assert generation["training_config"]["status"] == "completed"
        assert generation["metrics"]["timesteps_trained"] == 1

        model_path = Path(generation["model_path"])
        assert model_path.exists()
        loaded = NewsSentimentTrainer(model_id=model_id)
        loaded.load(str(model_path))
        assert loaded.is_trained


def test_news_sentiment_dataset_uses_canonical_news_labels(tmp_path: Path) -> None:
    """Dataset factory builds features and targets from canonical news rows."""

    config = APIConfig(api_key="training-secret-key", debug=True)
    model_service = _build_model_service(tmp_path)
    start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    end = start + timedelta(minutes=3)
    model = model_service.get_model(
        _create_supporting_ml_model(
            model_service,
            training_data_config={
                "symbols": ["AAPL"],
                "start_time": start.isoformat(),
                "end_time": end.isoformat(),
                "data_frequency": "1m",
                "market": {"provider": "file"},
                "news": {
                    "enabled": True,
                    "provider": "mock",
                    "include_body": True,
                    "limit": 4,
                },
            },
        )
    )
    data_service = create_default_data_acquisition_service(
        api_config=config,
        project_root=tmp_path,
    )

    features, targets = build_news_sentiment_dataset(
        model=model,
        data_config={},
        data_service=data_service,
        model_service=model_service,
        project_root=tmp_path,
    )

    assert len(features) == len(targets) > 0
    assert all("AAPL" in str(feature) for feature in features)
    assert all(-1.0 <= float(target) <= 1.0 for target in targets)


def test_news_sentiment_dataset_rejects_unlabeled_news_cache(
    tmp_path: Path,
) -> None:
    """News-derived datasets fail clearly when cached records lack labels."""

    config = APIConfig(api_key="training-secret-key", debug=True)
    model_service = _build_model_service(tmp_path)
    start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    end = start + timedelta(minutes=1)
    model = model_service.get_model(
        _create_supporting_ml_model(
            model_service,
            training_data_config={
                "symbols": ["AAPL"],
                "start_time": start.isoformat(),
                "end_time": end.isoformat(),
                "data_frequency": "1m",
                "cache_policy": "require_cache",
                "market": {"provider": "file"},
                "news": {"enabled": True, "provider": "mock"},
            },
        )
    )
    store = LocalDataStore(root=tmp_path / "data" / "sourced")
    store.write_news_records(
        [
            NewsRecord(
                timestamp=start,
                headline="Apple shares hold steady",
                body="No provider label was supplied for this story.",
                source="mock",
                symbols=["AAPL"],
                categories=["equities"],
                sentiment_score=None,
                url="https://example.test/aapl-unlabeled",
                news_id="mock-aapl-unlabeled",
            )
        ],
        provider="mock",
        symbol="AAPL",
        requested_start=start,
        requested_end=end,
    )
    data_service = create_default_data_acquisition_service(
        api_config=config,
        project_root=tmp_path,
        store=store,
    )

    with pytest.raises(TrainingFactoryError, match="sentiment_score labels"):
        build_news_sentiment_dataset(
            model=model,
            data_config={},
            data_service=data_service,
            model_service=model_service,
            project_root=tmp_path,
        )


def test_default_service_marks_generation_failed_for_missing_market_data(
    tmp_path: Path,
) -> None:
    """Required data failures surface as failed jobs and failed generations."""

    config = APIConfig(
        api_key="training-secret-key",
        debug=True,
        cors_origins=["http://localhost:3000"],
    )
    app = create_app(config)

    model_service = _build_model_service(tmp_path)
    app.state.model_service = model_service
    missing_path = tmp_path / "raw" / "missing-spy.csv"
    start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    end = start + timedelta(minutes=10)
    model = model_service.create_model(
        ModelConfigCreate(
            name="Missing Data PPO",
            model_type=ModelType.CORE_RL,
            trainer_type="stable_baselines3",
            algorithm="ppo",
            hyperparameters={
                "n_steps": 4,
                "batch_size": 4,
                "n_epochs": 1,
                "model_policy": "MultiInputPolicy",
            },
            training_data_config={
                "symbols": ["SPY"],
                "start_time": start.isoformat(),
                "end_time": end.isoformat(),
                "data_frequency": "1m",
                "market": {
                    "provider": "file",
                    "explicit_files": {"SPY": str(missing_path)},
                },
            },
            supporting_model_ids=[],
            strategy_ids=[],
            environment_config={"observation_window": 4},
            reward_function="profit_seeker",
        )
    )
    app.state.training_service = (
        training_service_module.create_default_training_service(
            ws_manager=app.state.ws_manager,
            model_service=model_service,
            project_root=tmp_path,
            api_config=config,
        )
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/training/jobs",
            headers=_auth_headers(),
            json={"model_id": model.model_id, "total_timesteps": 8},
        )
        assert response.status_code == 201, response.text
        job_id = response.json()["data"]["job_id"]

        failed = _wait_for_status(
            client,
            job_id,
            TrainingJobStatus.FAILED.value,
            timeout=10.0,
        )

        assert failed["started_at"] is not None
        assert failed["completed_at"] is not None
        assert failed["generation_id"] is not None
        assert "does not exist" in failed["error_message"]

        generation_response = client.get(
            f"/api/v1/generations/{failed['generation_id']}",
            headers=_auth_headers(),
        )
        assert generation_response.status_code == 200, generation_response.text
        generation = generation_response.json()["data"]
        assert generation["training_config"]["status"] == "failed"
        assert "does not exist" in generation["training_config"]["notes"]


@pytest.mark.requires_ib
@pytest.mark.slow
def test_default_service_trains_core_rl_after_ib_acquisition(
    confirm_ib_gateway: dict[str, object],
    tmp_path: Path,
) -> None:
    """Acquire live IB bars first, then train through default API runtime."""

    config = APIConfig(
        api_key="training-secret-key",
        debug=True,
        cors_origins=["http://localhost:3000"],
    )
    app = create_app(config)
    model_service = _build_model_service(tmp_path)
    app.state.model_service = model_service

    start = datetime(2026, 2, 13, 15, 0, tzinfo=UTC)
    end = start + timedelta(minutes=5)
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
        training_data_config,
        {"cache_policy": "refresh"},
    )
    acquirer = HistoricalMarketDataAcquirer(
        store=store,
        broker_registry=BrokerRegistry.isolated(),
        broker_config={},
        broker_connection_params={
            "host": str(confirm_ib_gateway["host"]),
            "port": int(confirm_ib_gateway["port"]),
            "client_id": int(confirm_ib_gateway["client_id"]) + 72,
        },
    )
    acquisition_result = acquirer.acquire(acquisition_request)
    if not acquisition_result.reports or acquisition_result.reports[0].row_count == 0:
        pytest.fail("IB acquisition returned no bars for provider-backed training")

    model = model_service.create_model(
        ModelConfigCreate(
            name="IB Backed PPO",
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
    app.state.training_service = (
        training_service_module.create_default_training_service(
            ws_manager=app.state.ws_manager,
            model_service=model_service,
            project_root=tmp_path,
            api_config=config,
        )
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/training/jobs",
            headers=_auth_headers(),
            json={"model_id": model.model_id, "total_timesteps": 8},
        )
        assert response.status_code == 201, response.text
        completed = _wait_for_status(
            client,
            response.json()["data"]["job_id"],
            TrainingJobStatus.COMPLETED.value,
            timeout=30.0,
        )

        assert completed["generation_id"] is not None
        generation_response = client.get(
            f"/api/v1/generations/{completed['generation_id']}",
            headers=_auth_headers(),
        )
        assert generation_response.status_code == 200, generation_response.text
        model_path = Path(generation_response.json()["data"]["model_path"])
        assert model_path.exists()


def test_cancel_completed_job_returns_409(
    api_client: tuple[TestClient, ModelService, _DeterministicExecutor],
) -> None:
    """Cancelling a job already in a terminal state returns a state error."""

    client, model_service, _executor = api_client
    model_id = _create_core_model(model_service)

    response = client.post(
        "/api/v1/training/jobs",
        headers=_auth_headers(),
        json={"model_id": model_id, "total_timesteps": 25},
    )
    assert response.status_code == 201
    job_id = response.json()["data"]["job_id"]
    _wait_for_status(client, job_id, TrainingJobStatus.COMPLETED.value)

    cancel_response = client.post(
        f"/api/v1/training/jobs/{job_id}/cancel",
        headers=_auth_headers(),
    )
    assert cancel_response.status_code == 409
    assert cancel_response.json()["error_code"] == "TRAINING_JOB_STATE_ERROR"


def test_cancel_running_job_transitions_to_cancelled(tmp_path: Path) -> None:
    """Cancelling a running job should eventually mark it CANCELLED."""

    config = APIConfig(
        api_key="training-secret-key",
        debug=True,
        cors_origins=["http://localhost:3000"],
    )
    app = create_app(config)

    model_service = _build_model_service(tmp_path)
    app.state.model_service = model_service

    worker = TrainingWorker(executor=_BlockingExecutor())
    training_service = TrainingService(
        ws_manager=app.state.ws_manager,
        model_service=model_service,
        generation_tracker=model_service.generation_tracker,
        worker=worker,
        jobs_path=tmp_path / "training_jobs.json",
    )
    app.state.training_service = training_service

    model_id = _create_core_model(model_service)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/training/jobs",
            headers=_auth_headers(),
            json={"model_id": model_id, "total_timesteps": 50},
        )
        assert response.status_code == 201
        job_id = response.json()["data"]["job_id"]

        _wait_for_status(client, job_id, TrainingJobStatus.RUNNING.value)

        cancel_response = client.post(
            f"/api/v1/training/jobs/{job_id}/cancel",
            headers=_auth_headers(),
        )
        assert cancel_response.status_code == 200

        cancelled = _wait_for_status(
            client,
            job_id,
            TrainingJobStatus.CANCELLED.value,
        )
        assert cancelled["completed_at"] is not None


def test_cancel_unknown_job_returns_404(
    api_client: tuple[TestClient, ModelService, _DeterministicExecutor],
) -> None:
    """Cancelling a missing job returns a not-found error."""

    client, _service, _executor = api_client
    response = client.post(
        "/api/v1/training/jobs/missing/cancel",
        headers=_auth_headers(),
    )
    assert response.status_code == 404
    assert response.json()["error_code"] == "TRAINING_JOB_NOT_FOUND"


def test_reorder_queued_jobs_returns_new_order(tmp_path: Path) -> None:
    """Queued jobs can be reordered through the training API."""

    config = APIConfig(
        api_key="training-secret-key",
        debug=True,
        cors_origins=["http://localhost:3000"],
    )
    app = create_app(config)

    model_service = _build_model_service(tmp_path)
    app.state.model_service = model_service

    worker = TrainingWorker(executor=_BlockingExecutor())
    training_service = TrainingService(
        ws_manager=app.state.ws_manager,
        model_service=model_service,
        generation_tracker=model_service.generation_tracker,
        worker=worker,
        jobs_path=tmp_path / "training_jobs.json",
    )
    app.state.training_service = training_service

    model_id = _create_core_model(model_service)

    with TestClient(app) as client:
        busy_response = client.post(
            "/api/v1/training/jobs",
            headers=_auth_headers(),
            json={"model_id": model_id, "total_timesteps": 50},
        )
        assert busy_response.status_code == 201
        busy_job_id = busy_response.json()["data"]["job_id"]
        _wait_for_status(client, busy_job_id, TrainingJobStatus.RUNNING.value)

        first_response = client.post(
            "/api/v1/training/jobs",
            headers=_auth_headers(),
            json={"model_id": model_id, "total_timesteps": 50},
        )
        second_response = client.post(
            "/api/v1/training/jobs",
            headers=_auth_headers(),
            json={"model_id": model_id, "total_timesteps": 75},
        )
        assert first_response.status_code == 201
        assert second_response.status_code == 201
        first_job_id = first_response.json()["data"]["job_id"]
        second_job_id = second_response.json()["data"]["job_id"]

        reorder_response = client.post(
            "/api/v1/training/jobs/reorder",
            headers=_auth_headers(),
            json={"job_ids": [second_job_id, first_job_id]},
        )

        assert reorder_response.status_code == 200
        assert [job["job_id"] for job in reorder_response.json()["data"]] == [
            second_job_id,
            first_job_id,
        ]

        queued_response = client.get(
            "/api/v1/training/jobs",
            headers=_auth_headers(),
            params={"status": TrainingJobStatus.QUEUED.value},
        )
        assert queued_response.status_code == 200
        assert [job["job_id"] for job in queued_response.json()["data"]] == [
            second_job_id,
            first_job_id,
        ]

        cancel_response = client.post(
            f"/api/v1/training/jobs/{busy_job_id}/cancel",
            headers=_auth_headers(),
        )
        assert cancel_response.status_code == 200


def test_websocket_receives_training_progress(
    api_client: tuple[TestClient, ModelService, _DeterministicExecutor],
) -> None:
    """A WebSocket client subscribed to a model receives training updates."""

    client, model_service, _executor = api_client
    model_id = _create_core_model(model_service)

    received: list[dict[str, Any]] = []
    with client.websocket_connect(
        f"/ws?api_key=training-secret-key&client_id=test-client"
    ) as websocket:
        # Skip the connection-acknowledgement message.
        websocket.receive_json()

        websocket.send_json({"type": "subscribe", "topic": f"training:{model_id}"})
        ack = websocket.receive_json()
        assert ack["type"] == "subscribed"

        response = client.post(
            "/api/v1/training/jobs",
            headers=_auth_headers(),
            json={"model_id": model_id, "total_timesteps": 30},
        )
        assert response.status_code == 201
        job_id = response.json()["data"]["job_id"]

        # Drain a few messages until we see a completion or progress payload.
        deadline = time.time() + 5.0
        statuses: list[str] = []
        while time.time() < deadline:
            try:
                message = websocket.receive_json(mode="text")
            except Exception:
                break
            received.append(message)
            data = message.get("data") or {}
            status_value = data.get("status")
            if status_value:
                statuses.append(status_value)
            if status_value == TrainingJobStatus.COMPLETED.value:
                break

        assert TrainingJobStatus.COMPLETED.value in statuses
        assert any((msg.get("data") or {}).get("job_id") == job_id for msg in received)
