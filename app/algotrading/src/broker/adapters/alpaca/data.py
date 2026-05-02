"""Historical market-data helpers for the Alpaca adapter."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Mapping

from algotrading.src.broker.adapters.alpaca.auth import AlpacaAuth
from algotrading.src.broker.adapters.alpaca.models import (
    alpaca_bar_to_bar_data,
    contract_to_alpaca_symbol,
)
from algotrading.src.broker.models import BarData, ContractSpec, InstrumentType


@dataclass(frozen=True)
class ParsedTimeFrame:
    """Broker-independent representation of an Alpaca bar timeframe."""

    amount: int
    unit: str

    def to_sdk_timeframe(self) -> object:
        """Convert this timeframe to an ``alpaca-py`` TimeFrame object."""

        try:
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        except ImportError as exc:
            raise RuntimeError(
                "alpaca-py is required to build Alpaca timeframe objects"
            ) from exc

        unit = _resolve_sdk_timeframe_unit(TimeFrameUnit, self.unit)
        return TimeFrame(self.amount, unit)


class AlpacaDataClient:
    """Wrapper around Alpaca's historical stock data SDK client."""

    def __init__(self, auth: AlpacaAuth, data_client: object | None = None) -> None:
        """Initialize a data wrapper with optional injected SDK client."""

        self.auth = auth
        self._client = data_client

    @property
    def client(self) -> object:
        """Return the underlying SDK data client, creating it lazily."""

        if self._client is None:
            self._client = self.auth.get_data_client()
        return self._client

    def get_historical_bars(
        self,
        contract: ContractSpec,
        end_datetime: datetime,
        duration: str,
        bar_size: str,
        data_type: str = "TRADES",
    ) -> list[BarData]:
        """Fetch historical bars and normalize them as ``BarData``."""

        if data_type.upper() != "TRADES":
            raise ValueError("Alpaca historical bars support data_type='TRADES' only")
        if contract.instrument_type not in {InstrumentType.STOCK, InstrumentType.INDEX}:
            raise ValueError(
                "Alpaca stock historical data supports stock/index symbols"
            )

        symbol = contract_to_alpaca_symbol(contract)
        end = _ensure_timezone(end_datetime)
        start = parse_duration_to_start(end, duration)
        timeframe = parse_bar_size(bar_size)
        request = self._build_stock_bars_request(symbol, start, end, timeframe)
        response = self.client.get_stock_bars(request)
        return [
            alpaca_bar_to_bar_data(bar)
            for bar in _extract_symbol_bars(response, symbol)
        ]

    def _build_stock_bars_request(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: ParsedTimeFrame,
    ) -> object:
        try:
            from alpaca.data.requests import StockBarsRequest
        except ImportError as exc:
            raise RuntimeError(
                "alpaca-py is required for Alpaca historical data requests"
            ) from exc

        return StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe.to_sdk_timeframe(),
            start=start,
            end=end,
            feed=_coerce_data_feed(self.auth.data_feed),
        )


def parse_duration_to_start(end_datetime: datetime, duration: str) -> datetime:
    """Convert an IB-style duration string into a start timestamp."""

    end = _ensure_timezone(end_datetime)
    amount, unit = _parse_amount_unit(duration)
    unit_upper = unit.upper()

    if unit_upper in {"S", "SEC", "SECS", "SECOND", "SECONDS"}:
        delta = timedelta(seconds=amount)
    elif unit_upper in {"MIN", "MINS", "MINUTE", "MINUTES"}:
        delta = timedelta(minutes=amount)
    elif unit_upper in {"H", "HOUR", "HOURS"}:
        delta = timedelta(hours=amount)
    elif unit_upper in {"D", "DAY", "DAYS"}:
        delta = timedelta(days=amount)
    elif unit_upper in {"W", "WEEK", "WEEKS"}:
        delta = timedelta(weeks=amount)
    elif unit_upper in {"M", "MONTH", "MONTHS"}:
        delta = timedelta(days=30 * amount)
    elif unit_upper in {"Y", "YEAR", "YEARS"}:
        delta = timedelta(days=365 * amount)
    else:
        raise ValueError(f"Unsupported duration unit: {unit}")
    return end - delta


def parse_bar_size(bar_size: str) -> ParsedTimeFrame:
    """Parse an IB-style bar-size string into an Alpaca timeframe."""

    amount, unit = _parse_amount_unit(bar_size)
    unit_upper = unit.upper()
    if unit_upper in {"S", "SEC", "SECS", "SECOND", "SECONDS"}:
        return ParsedTimeFrame(amount=amount, unit="Second")
    if unit_upper in {"MIN", "MINS", "MINUTE", "MINUTES"}:
        return ParsedTimeFrame(amount=amount, unit="Minute")
    if unit_upper in {"H", "HOUR", "HOURS"}:
        return ParsedTimeFrame(amount=amount, unit="Hour")
    if unit_upper in {"D", "DAY", "DAYS"}:
        return ParsedTimeFrame(amount=amount, unit="Day")
    if unit_upper in {"W", "WEEK", "WEEKS"}:
        return ParsedTimeFrame(amount=amount, unit="Week")
    if unit_upper in {"M", "MONTH", "MONTHS"}:
        return ParsedTimeFrame(amount=amount, unit="Month")
    raise ValueError(f"Unsupported bar size unit: {unit}")


def _extract_symbol_bars(response: object, symbol: str) -> list[object]:
    data = response
    if hasattr(response, "data"):
        data = getattr(response, "data")
    if isinstance(data, Mapping):
        symbol_bars = data.get(symbol) or data.get(symbol.upper()) or []
        return list(symbol_bars)
    if isinstance(data, list):
        return list(data)
    if hasattr(response, "df"):
        dataframe = getattr(response, "df")
        if dataframe is not None and hasattr(dataframe, "to_dict"):
            records = dataframe.reset_index().to_dict("records")
            return [
                record for record in records if record.get("symbol", symbol) == symbol
            ]
    return []


def _resolve_sdk_timeframe_unit(timeframe_unit: object, unit_name: str) -> object:
    candidate_names = [unit_name, unit_name.upper(), unit_name.lower(), unit_name[:3]]
    for candidate in candidate_names:
        if hasattr(timeframe_unit, candidate):
            return getattr(timeframe_unit, candidate)

    for member in timeframe_unit:
        if str(getattr(member, "name", "")).lower() == unit_name.lower():
            return member
        if str(getattr(member, "value", "")).lower() == unit_name.lower():
            return member
    raise ValueError(f"Alpaca SDK does not expose TimeFrameUnit.{unit_name}")


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


def _parse_amount_unit(value: str) -> tuple[int, str]:
    match = re.fullmatch(r"\s*(\d+)\s*([A-Za-z]+)\s*", value)
    if not match:
        raise ValueError(f"Expected '<amount> <unit>' format, got {value!r}")
    amount = int(match.group(1))
    if amount <= 0:
        raise ValueError("Time amount must be greater than zero")
    return amount, match.group(2)


def _ensure_timezone(value: datetime) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


__all__ = [
    "AlpacaDataClient",
    "ParsedTimeFrame",
    "parse_bar_size",
    "parse_duration_to_start",
]
