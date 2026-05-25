"""Smoke tests for the manual install and launch path."""

from __future__ import annotations

from algotrading.api.config import APIConfig
from algotrading.api.main import create_app
from algotrading.src.config.validation import validate_configuration
from fastapi.testclient import TestClient


def test_manual_install_configuration_validates_with_minimal_settings() -> None:
    """Minimal manual-install settings should satisfy startup validation."""

    settings = APIConfig(
        _env_file=None,
        api_key="manual-install-smoke-secret",
        environment="development",
        default_broker="interactive_brokers",
        ib_host="127.0.0.1",
        ib_port=7497,
    )

    is_valid, errors = validate_configuration(settings)

    assert is_valid is True
    assert errors == []


def test_manual_install_api_starts_in_process() -> None:
    """The FastAPI factory should start and expose launch health endpoints."""

    config = APIConfig(
        _env_file=None,
        api_key="manual-install-smoke-secret",
        environment="development",
        debug=True,
    )
    app = create_app(config)

    with TestClient(app) as client:
        health_response = client.get("/health")
        metrics_response = client.get("/metrics")

    assert health_response.status_code == 200
    assert health_response.json()["status"] == "healthy"
    assert metrics_response.status_code == 200
    assert "http_requests_total" in metrics_response.text
