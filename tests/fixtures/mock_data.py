"""Mock data generators for tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NewsTextItem:
    """Container for mock news text entries."""

    headline: str
    summary: str


def _ensure_positive(value: float, minimum: float = 0.01) -> float:
    """Clamp values to a small positive minimum for price calculations."""

    return value if value > minimum else minimum


def generate_ohlcv_data(
    num_bars: int,
    frequency: str,
    start_date: datetime,
    base_price: float,
) -> pd.DataFrame:
    """Generate a realistic OHLCV data frame using a random-walk process.

    Args:
        num_bars: Number of bars to generate.
        frequency: Pandas-compatible frequency string (e.g., "5s", "1min").
        start_date: Starting datetime for the first bar.
        base_price: Starting price for the first bar.

    Returns:
        DataFrame containing columns: date, open, high, low, close, volume, wap, count.

    Raises:
        ValueError: If num_bars is non-positive or base_price is not positive.
    """

    if num_bars <= 0:
        raise ValueError("num_bars must be a positive integer")
    if base_price <= 0:
        raise ValueError("base_price must be a positive number")

    rng = np.random.default_rng(42)
    dates = pd.date_range(start=start_date, periods=num_bars, freq=frequency)

    returns = rng.normal(loc=0.0, scale=0.002, size=num_bars)
    close = base_price * np.exp(np.cumsum(returns))

    open_prices = np.empty(num_bars)
    open_prices[0] = base_price
    open_prices[1:] = close[:-1]

    high_spread = np.abs(rng.normal(loc=0.0, scale=0.0015, size=num_bars))
    low_spread = np.abs(rng.normal(loc=0.0, scale=0.0015, size=num_bars))

    high = np.maximum(open_prices, close) * (1.0 + high_spread)
    low = np.minimum(open_prices, close) * (1.0 - low_spread)

    high = np.vectorize(_ensure_positive)(high)
    low = np.vectorize(_ensure_positive)(low)

    volatility = np.abs(close - open_prices) / np.maximum(open_prices, 0.01)
    base_volume = 50_000
    volume = (
        base_volume * (1.0 + volatility * 50.0) * rng.uniform(0.8, 1.2, size=num_bars)
    )
    volume = np.maximum(volume, 1.0).astype(int)

    wap = (high + low + close) / 3.0
    count = np.maximum((volume / 100).astype(int), 1)

    data = {
        "date": dates,
        "open": open_prices,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "wap": wap,
        "count": count,
    }

    return pd.DataFrame(data)


def generate_news_text_data(num_items: int) -> list[NewsTextItem]:
    """Generate deterministic mock news text items.

    Args:
        num_items: Number of news items to generate.

    Returns:
        List of NewsTextItem entries containing a headline and summary.

    Raises:
        ValueError: If num_items is non-positive.
    """

    if num_items <= 0:
        raise ValueError("num_items must be a positive integer")

    topics: Iterable[str] = (
        "earnings",
        "guidance",
        "macro",
        "product",
        "regulatory",
        "analyst",
    )
    rng = np.random.default_rng(7)
    topic_list = list(topics)

    items: list[NewsTextItem] = []
    for index in range(num_items):
        topic = topic_list[int(rng.integers(0, len(topic_list)))]
        change = rng.normal(loc=0.0, scale=1.0)
        direction = "up" if change >= 0 else "down"
        headline = f"Company sentiment {direction} on {topic} update"
        summary = (
            f"Mock news item {index + 1}: {topic} signals sentiment {direction} "
            f"with magnitude {abs(change):.2f}."
        )
        items.append(NewsTextItem(headline=headline, summary=summary))

    return items
