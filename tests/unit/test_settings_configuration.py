"""Unit tests for production settings and secret validation."""

from __future__ import annotations

import logging

import pytest
from algotrading.src.config.settings import Settings
from algotrading.src.config.validation import (
    configuration_snapshot,
    log_configuration,
    mask_secret,
    validate_configuration,
)

_ALPACA_ENV_NAMES = (
    "ALGOTRADING_ALPACA_API_KEY",
    "ALGOTRADING_ALPACA_SECRET_KEY",
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "APCA_API_KEY_ID",
    "APCA_API_SECRET_KEY",
)


def _clear_alpaca_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove Alpaca credential aliases for deterministic validation tests."""

    for env_name in _ALPACA_ENV_NAMES:
        monkeypatch.delenv(env_name, raising=False)


def test_settings_loads_prefixed_and_legacy_secret_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settings should read both prefixed and existing legacy env names."""

    monkeypatch.setenv("ALGOTRADING_API_KEY", "unit-api-key")
    monkeypatch.setenv("ALPACA_API_KEY", "alpaca-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "alpaca-secret")
    monkeypatch.setenv("BENZINGA_API_KEY", "benzinga-key")
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "alpha-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    settings = Settings(_env_file=None)

    assert settings.api_key == "unit-api-key"
    assert settings.alpaca_api_key == "alpaca-key"
    assert settings.alpaca_secret_key == "alpaca-secret"
    assert settings.benzinga_api_key == "benzinga-key"
    assert settings.alphavantage_api_key == "alpha-key"
    assert settings.openai_api_key == "openai-key"


def test_settings_parses_cors_origins_from_json_or_csv() -> None:
    """CORS origins should accept JSON arrays and comma-delimited values."""

    json_settings = Settings(
        _env_file=None,
        api_key="api-key",
        cors_origins='["http://localhost:3000", "https://app.example.com"]',
    )
    csv_settings = Settings(
        _env_file=None,
        api_key="api-key",
        cors_origins="http://localhost:3000, https://app.example.com",
    )

    expected = ["http://localhost:3000", "https://app.example.com"]
    assert json_settings.cors_origins == expected
    assert csv_settings.cors_origins == expected


def test_settings_returns_broker_configs_without_logging() -> None:
    """Broker helper methods should expose adapter-ready config values."""

    settings = Settings(
        _env_file=None,
        api_key="api-key",
        alpaca_api_key="alpaca-key",
        alpaca_secret_key="alpaca-secret",
        alpaca_paper=False,
        alpaca_data_feed="sip",
    )

    alpaca_config = settings.get_broker_config("alpaca")
    ib_params = settings.get_broker_connection_params("interactive_brokers")

    assert alpaca_config == {
        "api_key": "alpaca-key",
        "secret_key": "alpaca-secret",
        "paper": False,
        "data_feed": "sip",
    }
    assert ib_params == {"host": "127.0.0.1", "port": 7497, "client_id": 1}


def test_validate_configuration_rejects_missing_broker_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup validation should require at least one configured broker path."""

    _clear_alpaca_env(monkeypatch)
    settings = Settings(_env_file=None, api_key="api-key", ib_host="")

    valid, errors = validate_configuration(settings)

    assert valid is False
    assert any("At least one broker" in error for error in errors)


def test_validate_configuration_rejects_unsafe_production_settings() -> None:
    """Production validation should reject debug mode and weak API keys."""

    settings = Settings(
        _env_file=None,
        api_key="short-production-key",
        environment="production",
        debug=True,
    )

    valid, errors = validate_configuration(settings)

    assert valid is False
    assert "Debug mode must be disabled in production" in errors
    assert any("at least 32 characters" in error for error in errors)


def test_validate_configuration_requires_alpaca_keys_for_alpaca_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Alpaca should require API credentials when selected as default broker."""

    _clear_alpaca_env(monkeypatch)
    settings = Settings(
        _env_file=None,
        api_key="api-key",
        default_broker="alpaca",
    )

    valid, errors = validate_configuration(settings)

    assert valid is False
    assert any("ALGOTRADING_DEFAULT_BROKER=alpaca" in error for error in errors)


def test_mask_secret_hides_empty_and_short_values() -> None:
    """Secret masking should never expose short or absent values."""

    assert mask_secret(None) == "****"
    assert mask_secret("") == "****"
    assert mask_secret("abcd") == "****"
    assert mask_secret("abcdef") == "abcd**"


def test_configuration_snapshot_and_logs_mask_secrets(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Startup diagnostics should include masked values, never raw secrets."""

    api_secret = "api-secret-value"
    openai_secret = "openai-secret-value"
    alpaca_secret = "alpaca-secret-value"
    settings = Settings(
        _env_file=None,
        api_key=api_secret,
        openai_api_key=openai_secret,
        alpaca_api_key="alpaca-key-value",
        alpaca_secret_key=alpaca_secret,
        benzinga_api_key="benzinga-secret-value",
        alphavantage_api_key="alpha-secret-value",
    )

    snapshot = configuration_snapshot(settings)
    caplog.set_level(logging.INFO, logger="algotrading.src.config.validation")
    log_configuration(settings)

    rendered_records = "\n".join(
        f"{record.getMessage()} {record.__dict__}" for record in caplog.records
    )

    assert snapshot["api_key"] == "api-************"
    assert api_secret not in rendered_records
    assert openai_secret not in rendered_records
    assert alpaca_secret not in rendered_records
    assert "api-************" in rendered_records
