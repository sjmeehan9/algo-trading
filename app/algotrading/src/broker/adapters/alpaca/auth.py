"""Authentication helpers for Alpaca broker clients."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class AlpacaAuth:
    """Container for Alpaca credentials and SDK client factories.

    Args:
        api_key: Alpaca API key identifier.
        secret_key: Alpaca API secret key.
        paper: Whether to use Alpaca paper trading endpoints.
        data_feed: Alpaca market-data feed name, commonly ``iex`` or ``sip``.
    """

    api_key: str
    secret_key: str
    paper: bool = True
    data_feed: str = "iex"

    PAPER_TRADING_BASE_URL: ClassVar[str] = "https://paper-api.alpaca.markets"
    LIVE_TRADING_BASE_URL: ClassVar[str] = "https://api.alpaca.markets"
    DATA_BASE_URL: ClassVar[str] = "https://data.alpaca.markets"
    STREAM_BASE_URL: ClassVar[str] = "wss://stream.data.alpaca.markets"

    def __post_init__(self) -> None:
        """Validate that credentials are present without exposing their values."""

        if not self.api_key.strip():
            raise ValueError("Alpaca API key is required")
        if not self.secret_key.strip():
            raise ValueError("Alpaca secret key is required")
        if not self.data_feed.strip():
            raise ValueError("Alpaca data feed must be non-empty")

    @classmethod
    def from_env(cls, paper: bool | None = None) -> "AlpacaAuth":
        """Create credentials from supported Alpaca environment variables.

        The adapter accepts the Phase 6 local names as well as Alpaca's common
        ``APCA_*`` names for easier interoperability with SDK examples.
        """

        api_key = _first_env(
            "ALPACA_API_KEY",
            "ALGOTRADING_ALPACA_API_KEY",
            "APCA_API_KEY_ID",
        )
        secret_key = _first_env(
            "ALPACA_SECRET_KEY",
            "ALGOTRADING_ALPACA_SECRET_KEY",
            "APCA_API_SECRET_KEY",
        )
        paper_mode = (
            paper
            if paper is not None
            else _parse_bool(
                _first_env("ALPACA_PAPER", "ALGOTRADING_ALPACA_PAPER"),
                default=True,
            )
        )
        data_feed = _first_env("ALPACA_DATA_FEED", "ALGOTRADING_ALPACA_DATA_FEED")
        return cls(
            api_key=api_key or "",
            secret_key=secret_key or "",
            paper=paper_mode,
            data_feed=data_feed or "iex",
        )

    @property
    def trading_base_url(self) -> str:
        """Return the REST trading base URL implied by paper/live mode."""

        if self.paper:
            return self.PAPER_TRADING_BASE_URL
        return self.LIVE_TRADING_BASE_URL

    def get_headers(self) -> dict[str, str]:
        """Return Alpaca REST authentication headers."""

        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }

    def get_trading_client(self) -> object:
        """Return an authenticated Alpaca trading SDK client."""

        try:
            from alpaca.trading.client import TradingClient
        except ImportError as exc:
            raise RuntimeError(
                "alpaca-py is required for Alpaca trading operations"
            ) from exc

        return TradingClient(self.api_key, self.secret_key, paper=self.paper)

    def get_data_client(self) -> object:
        """Return an authenticated Alpaca stock historical-data SDK client."""

        try:
            from alpaca.data.historical import StockHistoricalDataClient
        except ImportError as exc:
            raise RuntimeError(
                "alpaca-py is required for Alpaca data operations"
            ) from exc

        return StockHistoricalDataClient(self.api_key, self.secret_key)

    def get_stream_client(self) -> object:
        """Return an authenticated Alpaca stock market-data stream client."""

        try:
            from alpaca.data.live import StockDataStream
        except ImportError as exc:
            raise RuntimeError(
                "alpaca-py is required for Alpaca streaming operations"
            ) from exc

        feed = _coerce_data_feed(self.data_feed)
        try:
            return StockDataStream(self.api_key, self.secret_key, feed=feed)
        except TypeError:
            return StockDataStream(self.api_key, self.secret_key)


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _coerce_data_feed(feed_name: str) -> object:
    normalized = feed_name.strip().lower()
    try:
        from alpaca.data.enums import DataFeed
    except ImportError:
        return normalized

    for member in DataFeed:
        if member.value == normalized or member.name.lower() == normalized:
            return member
    return normalized


__all__ = ["AlpacaAuth"]
