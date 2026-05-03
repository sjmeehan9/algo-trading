"""Integration tests for application startup configuration validation."""

from __future__ import annotations

import logging

import pytest
from algotrading.api.config import APIConfig
from algotrading.api.main import create_app
from fastapi.testclient import TestClient


def test_app_starts_with_valid_runtime_configuration() -> None:
    """The API should start and serve health with valid settings."""

    app = create_app(APIConfig(_env_file=None, api_key="startup-secret-key"))
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_app_rejects_invalid_startup_configuration() -> None:
    """The API factory should fail fast when production settings are unsafe."""

    config = APIConfig(
        _env_file=None,
        api_key="short-key",
        environment="production",
        debug=True,
    )

    with pytest.raises(RuntimeError, match="Debug mode must be disabled"):
        create_app(config)


def test_app_startup_logs_masked_secrets(caplog: pytest.LogCaptureFixture) -> None:
    """Startup logging should not expose raw configured secrets."""

    api_secret = "startup-api-secret-value"
    openai_secret = "startup-openai-secret-value"
    alpaca_secret = "startup-alpaca-secret-value"
    caplog.set_level(logging.INFO)

    create_app(
        APIConfig(
            _env_file=None,
            api_key=api_secret,
            openai_api_key=openai_secret,
            alpaca_api_key="startup-alpaca-key-value",
            alpaca_secret_key=alpaca_secret,
            benzinga_api_key="startup-benzinga-secret-value",
        )
    )

    rendered_records = "\n".join(
        f"{record.getMessage()} {record.__dict__}" for record in caplog.records
    )
    assert api_secret not in rendered_records
    assert openai_secret not in rendered_records
    assert alpaca_secret not in rendered_records
    assert "star**********************" in rendered_records
