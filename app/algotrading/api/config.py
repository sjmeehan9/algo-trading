"""Configuration models for the algo-trading API service."""

from __future__ import annotations

import json
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class APIConfig(BaseSettings):
    """Runtime configuration for the FastAPI service.

    Environment variables use the ``ALGOTRADING_`` prefix.
    """

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = Field(..., min_length=1)
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    debug: bool = False
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_prefix="ALGOTRADING_",
        extra="ignore",
        enable_decoding=False,
    )

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, value: str) -> str:
        """Ensure the configured API key is not blank."""

        normalized = value.strip()
        if not normalized:
            raise ValueError("ALGOTRADING_API_KEY must not be empty.")
        return normalized

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: object) -> object:
        """Support list, JSON string list, or comma-delimited CORS origins."""

        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []

            if stripped.startswith("["):
                parsed = json.loads(stripped)
                if not isinstance(parsed, list):
                    raise ValueError(
                        "ALGOTRADING_CORS_ORIGINS JSON must decode to a list."
                    )
                return [str(origin).strip() for origin in parsed if str(origin).strip()]

            return [origin.strip() for origin in stripped.split(",") if origin.strip()]

        if isinstance(value, list):
            return [str(origin).strip() for origin in value if str(origin).strip()]

        return value


def as_public_config(config: APIConfig) -> dict[str, Any]:
    """Return non-sensitive configuration fields for diagnostics.

    Args:
        config: Resolved API configuration.

    Returns:
        Dictionary containing non-secret values safe for logs.
    """

    return {
        "api_host": config.api_host,
        "api_port": config.api_port,
        "cors_origins": config.cors_origins,
        "debug": config.debug,
        "log_level": config.log_level,
    }
