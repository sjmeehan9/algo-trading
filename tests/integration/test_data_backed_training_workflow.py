"""Phase 7.6 file-backed regression for the data-backed training workflow.

This suite proves that the fixed training workflow uses real data paths end to
end: a core RL model is created, the default training service acquires market
data into the canonical local store, trains through the real trainers, and
records a generation that carries a loadable model artifact. It also proves the
required-news path fails or passes for the right reason.

These tests are deliberately tiny (small windows, low timesteps) so they run
routinely in CI with no provider flags. Provider-gated acquisition is covered
in ``test_data_acquisition_market.py`` and ``test_data_acquisition_news.py``.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from algotrading.api.config import APIConfig
from algotrading.api.main import create_app
from algotrading.api.schemas.data_sources import (
    DataFrequency,
    MarketDataProvider,
    NewsDataProvider,
    normalize_training_data_request,
)
from algotrading.api.schemas.models import ModelConfigCreate, ModelType
from algotrading.api.schemas.training import TrainingJobStatus
from algotrading.api.services import ModelService
from algotrading.api.services import training_service as training_service_module
from algotrading.api.services.data_acquisition_service import (
    create_default_data_acquisition_service,
)
from algotrading.api.training.factories import build_core_rl_environment
from algotrading.src.data_pipeline.acquisition import (
    NewsDataAcquisitionError,
    model_requires_news,
)
from algotrading.src.data_pipeline.types import DataType
from algotrading.src.models.registry import (
    CustomStrategyRegistry,
    ModelEntryConfig,
    ModelState,
    SupportingModelRegistry,
)
from algotrading.src.models.signals import SignalType
from algotrading.src.models.tracking import GenerationTracker, JsonFileStorage
from algotrading.src.trainers.sb3_trainer import SB3Algorithm, StableBaselines3Trainer
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

_API_KEY = "training-workflow-secret-key"


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


def _write_market_csv(path: Path, rows: int = 36) -> tuple[datetime, datetime]:
    """Write deterministic minute bars for the file-backed lifecycle."""

    path.parent.mkdir(parents=True, exist_ok=True)
    start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    records: list[dict[str, object]] = []
    for index in range(rows):
        timestamp = start + timedelta(minutes=index)
        open_price = 100.0 + (index * 0.1)
        close_price = open_price + (0.3 if index % 3 == 0 else -0.12)
        records.append(
            {
                "timestamp": timestamp.isoformat(),
                "symbol": "SPY",
                "open": open_price,
                "high": open_price + 0.4,
                "low": open_price - 0.4,
                "close": close_price,
                "volume": 1000 + index,
                "vwap": (open_price + close_price) / 2,
                "trade_count": 10 + index,
            }
        )
    pd.DataFrame(records).to_csv(path, index=False)
    return start, start + timedelta(minutes=rows - 1)


def _training_data_config(
    market_path: Path,
    start: datetime,
    end: datetime,
    *,
    news: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a file-backed training data config for SPY minute bars."""

    config: dict[str, object] = {
        "symbols": ["SPY"],
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "data_frequency": "1m",
        "market": {
            "provider": "file",
            "explicit_files": {"SPY": str(market_path)},
        },
    }
    if news is not None:
        config["news"] = news
    return config


def _create_core_model(
    service: ModelService,
    training_data_config: dict[str, object],
    *,
    name: str = "Workflow PPO",
    supporting_model_ids: list[str] | None = None,
) -> str:
    """Persist a tiny core RL model and return its model id."""

    response = service.create_model(
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
            supporting_model_ids=supporting_model_ids or [],
            strategy_ids=[],
            environment_config={"observation_window": 4},
            reward_function="profit_seeker",
        )
    )
    return response.model_id


def _register_news_supporting_model(service: ModelService) -> str:
    """Register a NEWS_TEXT sentiment supporting model and mark it READY.

    A core RL model can only reference supporting models that are in the READY
    state, so the registered model is transitioned through the lifecycle state
    machine (registered -> loading -> loaded -> ready) before it is used.
    """

    model_id = "sentiment-news-workflow"
    registry = service.supporting_registry
    registry.register(
        ModelEntryConfig(
            model_id=model_id,
            model_type="ml",
            signal_type=SignalType.SENTIMENT,
            trainer_class=(
                "algotrading.src.models.supporting.sentiment."
                "news_sentiment.NewsSentimentTrainer"
            ),
            input_data_types=[DataType.NEWS_TEXT],
            input_frequency=DataFrequency.IRREGULAR,
        )
    )
    for state in (ModelState.LOADING, ModelState.LOADED, ModelState.READY):
        registry.set_state(model_id, state)
    return model_id


def _wait_for_status(
    client: TestClient,
    job_id: str,
    *terminal: str,
    timeout: float = 25.0,
) -> dict[str, Any]:
    """Poll the get-job endpoint until a terminal status is reached."""

    deadline = time.time() + timeout
    last: dict[str, Any] = {}
    while time.time() < deadline:
        response = client.get(
            f"/api/v1/training/jobs/{job_id}", headers=_auth_headers()
        )
        assert response.status_code == 200, response.text
        last = response.json()["data"]
        if last["status"] in terminal:
            return last
        time.sleep(0.05)
    raise AssertionError(
        f"Job {job_id} did not reach {terminal}; last={last.get('status')}"
    )


def _start_default_training(
    *,
    config: APIConfig,
    model_service: ModelService,
    project_root: Path,
) -> Any:
    """Build the default training service and a wired app with it attached."""

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
    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_file_backed_lifecycle_acquires_data_and_produces_artifact(
    tmp_path: Path,
) -> None:
    """Full lifecycle: model create -> acquire -> train -> generation artifact.

    Asserts observable behavior rather than only HTTP 201:

    * The canonical local market file and manifest exist after training.
    * The manifest records the provider, requested range, and row count.
    * The completed generation exposes a loadable model artifact path.
    """

    config = APIConfig(
        api_key=_API_KEY,
        debug=True,
        cors_origins=["http://localhost:3000"],
    )
    model_service = _build_model_service(tmp_path)

    market_path = tmp_path / "raw" / "spy.csv"
    start, end = _write_market_csv(market_path)
    training_data_config = _training_data_config(market_path, start, end)
    model_id = _create_core_model(model_service, training_data_config)

    app = _start_default_training(
        config=config,
        model_service=model_service,
        project_root=tmp_path,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/training/jobs",
            headers=_auth_headers(),
            json={"model_id": model_id, "total_timesteps": 8},
        )
        assert response.status_code == 201, response.text
        created = response.json()["data"]
        job_id = created["job_id"]

        # The 4.2 regression: a job must not stay permanently queued.
        assert created["status"] == TrainingJobStatus.QUEUED.value
        assert created["started_at"] is None
        assert created["generation_id"] is None
        assert app.state.training_service.worker.is_running

        completed = _wait_for_status(
            client,
            job_id,
            TrainingJobStatus.COMPLETED.value,
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
        assert generation["model_id"] == model_id
        assert generation["training_config"]["status"] == "completed"
        assert generation["metrics"]["timesteps_trained"] == 8

        model_path = Path(generation["model_path"])
        assert model_path.exists(), "training generation must save a model artifact"

    # Observable behavior 1: the canonical sourced store was populated.
    market_dir = tmp_path / "data" / "sourced" / "market" / "file" / "SPY" / "MINUTE_1"
    data_file = market_dir / "market.csv"
    manifest_file = market_dir / "manifest.json"
    assert data_file.exists(), "acquisition must persist canonical market.csv"
    assert manifest_file.exists(), "acquisition must persist a manifest"

    # Observable behavior 2: the manifest records provider, range, and counts.
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["provider"] == MarketDataProvider.FILE.value
    assert manifest["symbols"] == ["SPY"]
    assert manifest["record_count"] > 0
    assert manifest["requested_range"]["start"] is not None
    assert manifest["requested_range"]["end"] is not None
    assert manifest["actual_range"]["start"] is not None
    assert manifest["actual_range"]["end"] is not None

    # Observable behavior 3: the persisted CSV row count matches the manifest.
    persisted_rows = len(pd.read_csv(data_file))
    assert persisted_rows == manifest["record_count"]

    # Observable behavior 4: the saved artifact reloads against a real env built
    # from the same canonical file-backed data contract.
    reload_data_service = create_default_data_acquisition_service(
        api_config=config,
        project_root=tmp_path,
    )
    load_env = build_core_rl_environment(
        model=model_service.get_model(model_id),
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


def test_required_news_is_detected_and_acquired_through_default_service(
    tmp_path: Path,
) -> None:
    """The required-news runtime path acquires news into the canonical store.

    Core RL training with supporting inputs is gated by the training factory
    (signal-feature wiring is later-phase work), so the news-required behavior
    is exercised where it actually runs: ``DataAcquisitionService`` inspects the
    referenced supporting model, determines news is required, and persists
    canonical news for the right reason.
    """

    config = APIConfig(
        api_key=_API_KEY,
        debug=True,
        cors_origins=["http://localhost:3000"],
    )
    model_service = _build_model_service(tmp_path)

    market_path = tmp_path / "raw" / "spy.csv"
    start, end = _write_market_csv(market_path)
    supporting_id = _register_news_supporting_model(model_service)
    training_data_config = _training_data_config(
        market_path,
        start,
        end,
        news={
            "enabled": True,
            "provider": "mock",
            "include_body": True,
            "limit": 8,
        },
    )
    model_id = _create_core_model(
        model_service,
        training_data_config,
        name="News Backed PPO",
        supporting_model_ids=[supporting_id],
    )
    model = model_service.get_model(model_id)

    # The supporting registry drives the required-news decision.
    assert model_requires_news(model, model_service.supporting_registry) is True

    data_service = create_default_data_acquisition_service(
        api_config=config,
        project_root=tmp_path,
    )
    # The default FastAPI dependency wires the supporting registry from the
    # model service; mirror that wiring so required-news detection works.
    data_service._supporting_registry = model_service.supporting_registry

    request = normalize_training_data_request(model.training_data_config, {})
    result = data_service.ensure_news_data(request, model=model)

    assert result.provider == NewsDataProvider.MOCK.value
    assert result.reports
    assert result.reports[0].symbol == "SPY"
    assert result.reports[0].row_count > 0

    # Observable behavior: news was acquired into the canonical store.
    news_dir = tmp_path / "data" / "sourced" / "news" / "mock" / "SPY"
    news_file = news_dir / "news.jsonl"
    news_manifest = news_dir / "manifest.json"
    assert news_file.exists(), "required news must persist canonical news.jsonl"
    assert news_manifest.exists(), "required news must persist a manifest"

    manifest = json.loads(news_manifest.read_text(encoding="utf-8"))
    assert manifest["provider"] == NewsDataProvider.MOCK.value
    assert manifest["record_count"] == result.reports[0].row_count
    # The JSONL line count matches the manifest record count.
    jsonl_lines = [
        line for line in news_file.read_text(encoding="utf-8").splitlines() if line
    ]
    assert len(jsonl_lines) == manifest["record_count"]


def test_required_news_fails_for_the_right_reason_when_unsatisfiable(
    tmp_path: Path,
) -> None:
    """Required news with no usable provider fails loudly, not silently.

    With ``cache_policy=refresh``, an empty canonical store, and a request that
    marks news required while configuring ``provider=none``, the runtime news
    path raises a news-specific acquisition error and writes no news artifact.
    The error type is ``NewsDataAcquisitionError`` (not a generic crash) and the
    failure is deterministic regardless of which provider credentials exist.
    """

    config = APIConfig(
        api_key=_API_KEY,
        debug=True,
        cors_origins=["http://localhost:3000"],
    )
    data_service = create_default_data_acquisition_service(
        api_config=config,
        project_root=tmp_path,
    )

    start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    end = start + timedelta(minutes=2)
    request = normalize_training_data_request(
        {
            "symbols": ["SPY"],
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "data_frequency": "1m",
            "cache_policy": "refresh",
            "market": {"provider": "file"},
            "news": {"enabled": True, "required": True, "provider": "none"},
        }
    )

    with pytest.raises(NewsDataAcquisitionError, match="no news provider"):
        data_service.ensure_news_data(request)

    # No news artifact should have been written for the unsatisfiable provider.
    assert not (tmp_path / "data" / "sourced" / "news").exists()


def test_missing_market_data_fails_lifecycle_before_generation_completes(
    tmp_path: Path,
) -> None:
    """A missing market CSV fails the lifecycle for the right reason.

    The data-acquire step must block training when the configured file does not
    exist, surfacing a failed job and a failed generation rather than producing
    an artifact from absent data.
    """

    config = APIConfig(
        api_key=_API_KEY,
        debug=True,
        cors_origins=["http://localhost:3000"],
    )
    model_service = _build_model_service(tmp_path)

    missing_path = tmp_path / "raw" / "missing-spy.csv"
    start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    end = start + timedelta(minutes=10)
    training_data_config = _training_data_config(missing_path, start, end)
    model_id = _create_core_model(
        model_service,
        training_data_config,
        name="Missing Data PPO",
    )

    app = _start_default_training(
        config=config,
        model_service=model_service,
        project_root=tmp_path,
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/training/jobs",
            headers=_auth_headers(),
            json={"model_id": model_id, "total_timesteps": 8},
        )
        assert response.status_code == 201, response.text
        job_id = response.json()["data"]["job_id"]

        failed = _wait_for_status(
            client,
            job_id,
            TrainingJobStatus.FAILED.value,
        )
        assert failed["generation_id"] is not None
        assert "does not exist" in (failed["error_message"] or "")

        generation = client.get(
            f"/api/v1/generations/{failed['generation_id']}",
            headers=_auth_headers(),
        ).json()["data"]
        assert generation["training_config"]["status"] == "failed"

    # No canonical market data should have been persisted from a missing file.
    assert not (tmp_path / "data" / "sourced" / "market").exists()
