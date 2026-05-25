"""Application settings loaded from environment variables."""

from __future__ import annotations

import json
from functools import lru_cache

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central runtime settings for the trading application.

    Environment variables use the ``ALGOTRADING_`` prefix by default. Selected
    external-service credentials also accept the legacy unprefixed names already
    used by earlier phases, such as ``OPENAI_API_KEY`` and ``ALPACA_API_KEY``.
    """

    model_config = SettingsConfigDict(
        env_file=(".env", ".env/.env.local"),
        env_file_encoding="utf-8",
        env_prefix="ALGOTRADING_",
        case_sensitive=False,
        extra="ignore",
        enable_decoding=False,
        populate_by_name=True,
    )

    app_name: str = "Algo-Trading Hub"
    environment: str = Field(
        default="development",
        description="Runtime environment: development, staging, or production.",
    )
    debug: bool = Field(default=False, description="Enable local debug behavior.")
    log_level: str = Field(default="INFO", description="Python logging level.")

    api_host: str = Field(default="0.0.0.0", description="API server host.")
    api_port: int = Field(default=8000, description="API server port.")
    api_key: str = Field(
        ...,
        min_length=1,
        description="API authentication key.",
        validation_alias=AliasChoices("ALGOTRADING_API_KEY"),
    )
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000"],
        description="Allowed CORS origins.",
    )

    ib_host: str = Field(default="127.0.0.1", description="TWS/Gateway host.")
    ib_port: int = Field(
        default=7497,
        description="TWS/Gateway port: 7497 for paper, 7496 for live.",
    )
    ib_client_id: int = Field(default=1, description="Interactive Brokers client ID.")

    alpaca_api_key: str | None = Field(
        default=None,
        description="Alpaca API key.",
        validation_alias=AliasChoices(
            "ALGOTRADING_ALPACA_API_KEY",
            "ALPACA_API_KEY",
            "APCA_API_KEY_ID",
        ),
    )
    alpaca_secret_key: str | None = Field(
        default=None,
        description="Alpaca secret key.",
        validation_alias=AliasChoices(
            "ALGOTRADING_ALPACA_SECRET_KEY",
            "ALPACA_SECRET_KEY",
            "APCA_API_SECRET_KEY",
        ),
    )
    alpaca_paper: bool = Field(
        default=True,
        description="Use Alpaca paper trading endpoints.",
        validation_alias=AliasChoices("ALGOTRADING_ALPACA_PAPER", "ALPACA_PAPER"),
    )
    alpaca_data_feed: str = Field(
        default="iex",
        description="Alpaca market-data feed, commonly iex or sip.",
        validation_alias=AliasChoices(
            "ALGOTRADING_ALPACA_DATA_FEED",
            "ALPACA_DATA_FEED",
        ),
    )

    benzinga_api_key: str | None = Field(
        default=None,
        description="Benzinga News API key.",
        validation_alias=AliasChoices(
            "ALGOTRADING_BENZINGA_API_KEY",
            "BENZINGA_API_KEY",
        ),
    )
    alphavantage_api_key: str | None = Field(
        default=None,
        description="Alpha Vantage API key.",
        validation_alias=AliasChoices(
            "ALGOTRADING_ALPHAVANTAGE_API_KEY",
            "ALPHAVANTAGE_API_KEY",
        ),
    )
    news_provider: str = Field(
        default="benzinga",
        description="Primary news provider: benzinga or alphavantage.",
    )
    news_fallback_provider: str | None = Field(
        default="alphavantage",
        description="Fallback news provider, or empty to disable fallback.",
    )

    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key for LLM hyperparameter optimization.",
        validation_alias=AliasChoices("ALGOTRADING_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model name.")
    openai_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        description="OpenAI request timeout in seconds.",
    )

    data_path: str = Field(default="data", description="Data storage path.")
    model_path: str = Field(default="models", description="Model storage path.")
    log_path: str = Field(default="logs", description="Log storage path.")

    default_broker: str = Field(
        default="interactive_brokers",
        description="Default broker registry key.",
    )
    max_position_pct: float = Field(
        default=0.1,
        gt=0,
        le=1,
        description="Max position size as a fraction of portfolio value.",
    )
    min_confidence: float = Field(
        default=0.6,
        ge=0,
        le=1,
        description="Minimum trading decision confidence for execution.",
    )

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, value: str) -> str:
        """Normalize and validate the required API authentication key."""

        normalized = value.strip()
        if not normalized:
            raise ValueError("ALGOTRADING_API_KEY must not be empty.")
        return normalized

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, value: str) -> str:
        """Validate the runtime environment name."""

        normalized = value.strip().lower()
        valid = {"development", "staging", "production"}
        if normalized not in valid:
            raise ValueError(f"environment must be one of {sorted(valid)}")
        return normalized

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        """Normalize and validate the logging level."""

        normalized = value.strip().upper()
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalized not in valid:
            raise ValueError(f"log_level must be one of {sorted(valid)}")
        return normalized

    @field_validator("default_broker")
    @classmethod
    def normalize_default_broker(cls, value: str) -> str:
        """Normalize the configured default broker key."""

        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("default_broker must be non-empty")
        return normalized

    @field_validator("news_provider")
    @classmethod
    def validate_news_provider(cls, value: str) -> str:
        """Validate the primary news provider name."""

        normalized = value.strip().lower()
        valid = {"benzinga", "alphavantage"}
        if normalized not in valid:
            raise ValueError(f"news_provider must be one of {sorted(valid)}")
        return normalized

    @field_validator("news_fallback_provider", mode="before")
    @classmethod
    def normalize_optional_provider(cls, value: object) -> object:
        """Normalize empty fallback provider values to ``None``."""

        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("news_fallback_provider")
    @classmethod
    def validate_news_fallback_provider(cls, value: str | None) -> str | None:
        """Validate the optional fallback news provider name."""

        if value is None:
            return None
        normalized = value.strip().lower()
        valid = {"benzinga", "alphavantage"}
        if normalized not in valid:
            raise ValueError(f"news_fallback_provider must be one of {sorted(valid)}")
        return normalized

    @field_validator(
        "alpaca_api_key",
        "alpaca_secret_key",
        "benzinga_api_key",
        "alphavantage_api_key",
        "openai_api_key",
        mode="before",
    )
    @classmethod
    def normalize_optional_secret(cls, value: object) -> object:
        """Normalize blank optional secret values to ``None``."""

        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return value

    @field_validator("alpaca_data_feed")
    @classmethod
    def normalize_alpaca_data_feed(cls, value: str) -> str:
        """Normalize the Alpaca market-data feed name."""

        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("alpaca_data_feed must be non-empty")
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

    def get_broker_config(self, broker_name: str) -> dict[str, object]:
        """Return adapter configuration for a specific broker.

        The Alpaca result includes credentials for injection into the adapter.
        Callers must never log this dictionary.
        """

        normalized = broker_name.strip().lower()
        if normalized == "interactive_brokers":
            return {
                "host": self.ib_host,
                "port": self.ib_port,
                "client_id": self.ib_client_id,
                "paper_trading": self.ib_port != 7496,
            }
        if normalized == "alpaca":
            return {
                "api_key": self.alpaca_api_key,
                "secret_key": self.alpaca_secret_key,
                "paper": self.alpaca_paper,
                "data_feed": self.alpaca_data_feed,
            }
        return {}

    def get_broker_connection_params(self, broker_name: str) -> dict[str, object]:
        """Return connection parameters for a broker adapter."""

        normalized = broker_name.strip().lower()
        if normalized == "interactive_brokers":
            return {
                "host": self.ib_host,
                "port": self.ib_port,
                "client_id": self.ib_client_id,
            }
        return {}

    def is_production(self) -> bool:
        """Return whether these settings describe a production environment."""

        return self.environment == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings instance."""

    return Settings()


__all__ = ["Settings", "get_settings"]
