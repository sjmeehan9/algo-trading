"""Integration tests for Phase 5.3 training control API endpoints."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from algotrading.api.config import APIConfig
from algotrading.api.main import create_app
from algotrading.api.schemas.models import ModelConfigCreate, ModelType
from algotrading.api.schemas.training import TrainingJobStatus
from algotrading.api.services import ModelService
from algotrading.api.services.training_service import TrainingService
from algotrading.api.workers.training_worker import (
    TrainingCancelledError,
    TrainingExecutionResult,
    TrainingJobContext,
    TrainingProgressUpdate,
    TrainingWorker,
)
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
