"""Factory for constructing configured concrete news provider sources."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from algotrading.src.data_pipeline.sources.alphavantage_news import (
    AlphaVantageNewsSource,
)
from algotrading.src.data_pipeline.sources.benzinga_news import BenzingaNewsSource
from algotrading.src.data_pipeline.sources.exceptions import (
    DataSourceConnectionError,
    DataValidationError,
)
from algotrading.src.data_pipeline.sources.news_source import NewsDataSource
from algotrading.src.data_pipeline.sources.news_utils import resolve_env_placeholder

_ENV_FALLBACKS = {
    "benzinga": ("ALGOTRADING_BENZINGA_API_KEY", "BENZINGA_API_KEY"),
    "alphavantage": (
        "ALGOTRADING_ALPHAVANTAGE_API_KEY",
        "ALPHAVANTAGE_API_KEY",
    ),
}


class NewsSourceFactory:
    """Create configured primary and fallback news data source instances."""

    def __init__(
        self,
        provider_config: dict[str, Any],
        credentials_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize factory from provider and credential YAML payloads."""

        providers = provider_config.get("providers")
        if not isinstance(providers, dict) or not providers:
            raise DataValidationError(
                "provider_config must define non-empty 'providers'"
            )

        self._provider_config = provider_config
        self._credentials_config = credentials_config or {}

    @classmethod
    def from_files(
        cls,
        providers_path: str | Path,
        credentials_path: str | Path | None = None,
    ) -> NewsSourceFactory:
        """Create factory from YAML files on disk."""

        provider_config = cls._load_yaml(Path(providers_path))

        resolved_credentials_path: Path | None = None
        if credentials_path is not None:
            resolved_credentials_path = Path(credentials_path)
        else:
            default_path = Path(providers_path).parent / "news_credentials.yml"
            if default_path.exists():
                resolved_credentials_path = default_path

        credentials_config: dict[str, Any] | None = None
        if resolved_credentials_path is not None and resolved_credentials_path.exists():
            credentials_config = cls._load_yaml(resolved_credentials_path)

        return cls(
            provider_config=provider_config,
            credentials_config=credentials_config,
        )

    def create_primary(self) -> NewsDataSource:
        """Create the configured primary provider instance."""

        provider_name = self._provider_name("primary_provider")
        return self.create(provider_name)

    def create_fallback(self) -> NewsDataSource | None:
        """Create the configured fallback provider instance when present."""

        provider_name = self._provider_config.get("fallback_provider")
        if provider_name is None:
            return None

        provider_text = str(provider_name).strip().lower()
        if not provider_text:
            return None
        return self.create(provider_text)

    def create_with_fallback(self) -> tuple[NewsDataSource, NewsDataSource | None]:
        """Create primary and fallback providers in one call."""

        primary = self.create_primary()
        fallback = self.create_fallback()
        return primary, fallback

    def create(self, provider_name: str) -> NewsDataSource:
        """Create one configured provider by name."""

        normalized = provider_name.strip().lower()
        providers = self._provider_config["providers"]
        if normalized not in providers:
            raise DataValidationError(f"Unknown news provider '{provider_name}'")

        provider_payload = providers[normalized]
        if not isinstance(provider_payload, dict):
            raise DataValidationError(
                f"Provider config for '{normalized}' must be a mapping"
            )

        credential_payload = self._credentials_config.get(normalized, {})
        if credential_payload is None:
            credential_payload = {}
        if not isinstance(credential_payload, dict):
            raise DataValidationError(
                f"Credential config for '{normalized}' must be a mapping"
            )

        config = dict(provider_payload)
        config.update(
            {
                key: value
                for key, value in credential_payload.items()
                if key != "api_key"
            }
        )

        api_key = self._resolve_api_key(normalized, credential_payload)

        if normalized == "benzinga":
            return BenzingaNewsSource(api_key=api_key, config=config)
        if normalized == "alphavantage":
            return AlphaVantageNewsSource(api_key=api_key, config=config)

        raise DataValidationError(f"Unsupported news provider '{provider_name}'")

    def _provider_name(self, field_name: str) -> str:
        """Resolve and validate provider selection field values."""

        value = self._provider_config.get(field_name)
        provider_name = str(value).strip().lower()
        if not provider_name:
            raise DataValidationError(f"Missing required provider field '{field_name}'")
        return provider_name

    def _resolve_api_key(
        self,
        provider_name: str,
        credential_payload: dict[str, Any],
    ) -> str:
        """Resolve API key from credentials config or environment fallback."""

        key_value = credential_payload.get("api_key")
        if isinstance(key_value, str) and key_value.strip():
            try:
                resolved = resolve_env_placeholder(key_value)
            except ValueError as exc:
                raise DataSourceConnectionError(str(exc)) from exc
            if resolved:
                return resolved

        env_names = _ENV_FALLBACKS.get(provider_name)
        if env_names is None:
            raise DataValidationError(
                f"No API key resolution strategy for provider '{provider_name}'"
            )

        for env_name in env_names:
            env_value = os.getenv(env_name)
            if env_value is not None and env_value.strip():
                return env_value.strip()

        raise DataSourceConnectionError(
            f"Missing API key for provider '{provider_name}'. "
            f"Set one of {', '.join(env_names)} or populate news_credentials.yml."
        )

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        """Load and validate one YAML file into a dictionary payload."""

        if not path.exists():
            raise DataValidationError(f"Config file not found: {path}")

        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}

        if not isinstance(payload, dict):
            raise DataValidationError(f"YAML root must be a mapping: {path}")
        return payload


__all__ = ["NewsSourceFactory"]
