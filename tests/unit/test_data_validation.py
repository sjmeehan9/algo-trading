"""Tests for sample data validation utilities."""

from __future__ import annotations

import pandas as pd
import pytest

from tests.fixtures.data_validation import load_sample_data, validate_market_data


def test_validate_market_data_valid() -> None:
    """Verify valid sample data passes validation."""

    df = load_sample_data("sample_normal_day")
    is_valid, errors = validate_market_data(df)

    assert is_valid
    assert errors == []


def test_validate_market_data_invalid_high_low() -> None:
    """Verify invalid price relationships are detected."""

    df = load_sample_data("sample_normal_day").head(5).copy()
    df.loc[df.index[0], "high"] = df.loc[df.index[0], "open"] - 1.0

    is_valid, errors = validate_market_data(df)

    assert not is_valid
    assert any("High price below" in error for error in errors)


def test_validate_market_data_missing_columns() -> None:
    """Verify missing columns are reported."""

    df = load_sample_data("sample_normal_day")[["date", "open", "close"]].copy()
    is_valid, errors = validate_market_data(df)

    assert not is_valid
    assert any("Missing required columns" in error for error in errors)


def test_load_sample_data_missing_file() -> None:
    """Ensure missing sample files raise FileNotFoundError."""

    with pytest.raises(FileNotFoundError):
        load_sample_data("does_not_exist")


def test_load_sample_data_returns_dataframe() -> None:
    """Ensure loader returns a DataFrame with expected columns."""

    df = load_sample_data("sample_low_volume")
    assert isinstance(df, pd.DataFrame)
    assert {"date", "open", "high", "low", "close", "volume"}.issubset(df.columns)
