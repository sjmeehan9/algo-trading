"""Unit tests for API configuration loading and validation."""

from __future__ import annotations

import pytest
from algotrading.api.config import APIConfig
from pydantic import ValidationError


def test_api_config_loads_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """APIConfig should read expected values from environment variables."""

    monkeypatch.setenv("ALGOTRADING_API_KEY", "unit-test-key")
    monkeypatch.setenv("ALGOTRADING_API_PORT", "9001")
    monkeypatch.setenv(
        "ALGOTRADING_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:5173",
    )

    config = APIConfig()

    assert config.api_host == "0.0.0.0"
    assert config.api_port == 9001
    assert config.api_key == "unit-test-key"
    assert config.cors_origins == ["http://localhost:3000", "http://127.0.0.1:5173"]


def test_api_config_supports_json_list_for_cors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CORS origins can be supplied as a JSON list string."""

    monkeypatch.setenv("ALGOTRADING_API_KEY", "json-cors-key")
    monkeypatch.setenv(
        "ALGOTRADING_CORS_ORIGINS",
        '["http://localhost:3000", "https://frontend.example.com"]',
    )

    config = APIConfig()

    assert config.cors_origins == [
        "http://localhost:3000",
        "https://frontend.example.com",
    ]


def test_api_config_requires_non_empty_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """APIConfig should fail validation when API key is missing."""

    monkeypatch.delenv("ALGOTRADING_API_KEY", raising=False)

    with pytest.raises(ValidationError):
        APIConfig()
