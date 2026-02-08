"""Sample data loading and validation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


def validate_market_data(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate market data integrity.

    Args:
        df: DataFrame containing market data.

    Returns:
        Tuple of (is_valid, error_messages).
    """

    errors: list[str] = []
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
        return False, errors

    if df.empty:
        errors.append("DataFrame is empty")
        return False, errors

    price_columns: Iterable[str] = ("open", "high", "low", "close")
    if df[list(price_columns)].isna().any().any():
        errors.append("NaN values detected in price columns")

    if df["volume"].isna().any():
        errors.append("NaN values detected in volume column")

    invalid_high = df["high"] < df[["open", "close"]].max(axis=1)
    if invalid_high.any():
        errors.append("High price below open/close detected")

    invalid_low = df["low"] > df[["open", "close"]].min(axis=1)
    if invalid_low.any():
        errors.append("Low price above open/close detected")

    if (df["volume"] < 0).any():
        errors.append("Negative volume detected")

    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.isna().any():
        errors.append("Invalid date values detected")
    elif not dates.is_monotonic_increasing:
        errors.append("Dates are not monotonically increasing")

    return len(errors) == 0, errors


def load_sample_data(name: str) -> pd.DataFrame:
    """Load a sample data CSV by filename stem.

    Args:
        name: Filename stem (e.g., "sample_normal_day").

    Returns:
        DataFrame containing the sample data.

    Raises:
        FileNotFoundError: If the sample data file does not exist.
    """

    data_dir = Path(__file__).resolve().parent / "data"
    file_path = data_dir / f"{name}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Sample data file not found: {file_path}")

    return pd.read_csv(file_path)
