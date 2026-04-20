"""Unit tests for the news provider configuration template."""

from __future__ import annotations

from pathlib import Path

import yaml


def test_news_provider_config_matches_component_3_9_template() -> None:
    """The provider template exposes required keys from Component 3.9 spec."""

    config_path = (
        Path(__file__).resolve().parents[2]
        / "app"
        / "algotrading"
        / "config"
        / "news_providers.yml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["primary_provider"] == "benzinga"
    assert config["fallback_provider"] == "alphavantage"

    providers = config["providers"]
    assert set(providers.keys()) == {"benzinga", "alphavantage"}

    benzinga = providers["benzinga"]
    assert benzinga["name"] == "Benzinga News API"
    assert benzinga["poll_interval_seconds"] == 10
    assert benzinga["provides_sentiment"] is False
    assert benzinga["required_credentials"] == ["api_key"]

    alphavantage = providers["alphavantage"]
    assert alphavantage["name"] == "Alpha Vantage News Sentiment"
    assert alphavantage["poll_interval_seconds"] == 60
    assert alphavantage["provides_sentiment"] is True
    assert alphavantage["required_credentials"] == ["api_key"]
