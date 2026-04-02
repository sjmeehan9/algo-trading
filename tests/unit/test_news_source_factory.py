"""Unit tests for configured news source factory."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from algotrading.src.data_pipeline.sources import (
    AlphaVantageNewsSource,
    BenzingaNewsSource,
    NewsSourceFactory,
)
from algotrading.src.data_pipeline.sources.exceptions import DataSourceConnectionError


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


def test_factory_creates_primary_and_fallback_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Factory should materialize configured benzinga primary and AV fallback."""

    providers_path = tmp_path / "news_providers.yml"
    credentials_path = tmp_path / "news_credentials.yml"

    _write_yaml(
        providers_path,
        """
        primary_provider: benzinga
        fallback_provider: alphavantage
        providers:
          benzinga:
            calls_per_minute: 100
            poll_interval_seconds: 10
          alphavantage:
            calls_per_minute: 5
            poll_interval_seconds: 60
        """,
    )
    _write_yaml(
        credentials_path,
        """
        benzinga:
          api_key: "${BENZINGA_API_KEY}"
        alphavantage:
          api_key: "${ALPHAVANTAGE_API_KEY}"
        """,
    )

    monkeypatch.setenv("BENZINGA_API_KEY", "benzinga-test")
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "av-test")

    factory = NewsSourceFactory.from_files(
        providers_path=providers_path,
        credentials_path=credentials_path,
    )

    primary = factory.create_primary()
    fallback = factory.create_fallback()

    assert isinstance(primary, BenzingaNewsSource)
    assert isinstance(fallback, AlphaVantageNewsSource)


def test_factory_raises_when_api_key_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Factory surfaces a connection error when API key resolution fails."""

    providers_path = tmp_path / "news_providers.yml"
    credentials_path = tmp_path / "news_credentials.yml"

    _write_yaml(
        providers_path,
        """
        primary_provider: benzinga
        providers:
          benzinga:
            calls_per_minute: 100
            poll_interval_seconds: 10
        """,
    )
    _write_yaml(
        credentials_path,
        """
        benzinga:
          api_key: "${MISSING_BENZINGA_KEY}"
        """,
    )

    monkeypatch.delenv("MISSING_BENZINGA_KEY", raising=False)
    monkeypatch.delenv("BENZINGA_API_KEY", raising=False)

    factory = NewsSourceFactory.from_files(
        providers_path=providers_path,
        credentials_path=credentials_path,
    )

    with pytest.raises(DataSourceConnectionError):
        factory.create_primary()
