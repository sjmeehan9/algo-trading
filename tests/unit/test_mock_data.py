"""Tests for mock data generators and fixtures."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from tests.fixtures.mock_data import generate_news_text_data, generate_ohlcv_data


def test_generate_ohlcv_data_relationships() -> None:
    """Ensure OHLCV relationships are valid in generated data."""

    data = generate_ohlcv_data(
        num_bars=250,
        frequency="5s",
        start_date=datetime(2024, 1, 2, 9, 30),
        base_price=150.0,
    )

    assert isinstance(data, pd.DataFrame)
    assert data.shape[0] == 250
    for _, row in data.iterrows():
        assert row["high"] >= max(row["open"], row["close"])
        assert row["low"] <= min(row["open"], row["close"])
        assert row["volume"] >= 0


def test_generate_news_text_data() -> None:
    """Verify mock news text generator returns expected length."""

    items = generate_news_text_data(5)
    assert len(items) == 5
    assert all(item.headline for item in items)
    assert all(item.summary for item in items)


def test_mock_market_data_fixture(mock_market_data: pd.DataFrame) -> None:
    """Confirm mock market data fixture returns the expected columns."""

    expected_columns = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "wap",
        "count",
    }
    assert expected_columns.issubset(set(mock_market_data.columns))
    assert len(mock_market_data) == 1000


def test_mock_pipeline_config_fixture(mock_pipeline_config: dict[str, object]) -> None:
    """Confirm mock pipeline config fixture returns a dictionary."""

    assert isinstance(mock_pipeline_config, dict)
    assert "pipeline" in mock_pipeline_config
