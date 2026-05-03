"""Configuration models for the algo-trading API service."""

from __future__ import annotations

from typing import Any

from algotrading.src.config.settings import Settings


class APIConfig(Settings):
    """Backward-compatible API configuration model.

    The API now uses the shared application ``Settings`` model while retaining
    the ``APIConfig`` import path used by Phase 5 services and tests.
    """


def as_public_config(config: Settings) -> dict[str, Any]:
    """Return non-sensitive configuration fields for diagnostics.

    Args:
        config: Resolved application configuration.

    Returns:
        Dictionary containing non-secret values safe for logs.
    """

    return {
        "app_name": config.app_name,
        "environment": config.environment,
        "api_host": config.api_host,
        "api_port": config.api_port,
        "cors_origins": config.cors_origins,
        "debug": config.debug,
        "log_level": config.log_level,
        "default_broker": config.default_broker,
        "alpaca_configured": bool(config.alpaca_api_key and config.alpaca_secret_key),
        "benzinga_configured": bool(config.benzinga_api_key),
        "alphavantage_configured": bool(config.alphavantage_api_key),
        "news_provider": config.news_provider,
        "news_fallback_provider": config.news_fallback_provider,
        "openai_configured": bool(config.openai_api_key),
        "openai_model": config.openai_model,
        "openai_timeout_seconds": config.openai_timeout_seconds,
    }


__all__ = ["APIConfig", "as_public_config"]
