"""Canonical data-source request schemas for training and backtesting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, time
from enum import Enum
from pathlib import Path
from typing import TypeAlias

from algotrading.src.broker.models import InstrumentType
from algotrading.src.data_pipeline import DataFrequency, NewsQuery
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RawConfig: TypeAlias = Mapping[str, object] | None


class DataSourceConfigError(ValueError):
    """Raised when a data-source payload cannot be normalized."""


class CachePolicy(str, Enum):
    """Supported cache policies for sourced historical data."""

    PREFER_CACHE = "prefer_cache"
    REFRESH = "refresh"
    REQUIRE_CACHE = "require_cache"


class MarketDataProvider(str, Enum):
    """Supported market-data providers for historical acquisition."""

    FILE = "file"
    IB = "ib"
    ALPACA = "alpaca"


class NewsDataProvider(str, Enum):
    """Supported historical news providers."""

    NONE = "none"
    MOCK = "mock"
    BENZINGA = "benzinga"
    ALPHAVANTAGE = "alphavantage"


_FREQUENCY_ALIASES: dict[str, DataFrequency] = {
    "tick": DataFrequency.TICK,
    "realtime": DataFrequency.TICK,
    "1s": DataFrequency.SECOND_1,
    "1 sec": DataFrequency.SECOND_1,
    "1 secs": DataFrequency.SECOND_1,
    "1 second": DataFrequency.SECOND_1,
    "1 seconds": DataFrequency.SECOND_1,
    "5s": DataFrequency.SECOND_5,
    "5 sec": DataFrequency.SECOND_5,
    "5 secs": DataFrequency.SECOND_5,
    "5 second": DataFrequency.SECOND_5,
    "5 seconds": DataFrequency.SECOND_5,
    "10s": DataFrequency.SECOND_10,
    "10 sec": DataFrequency.SECOND_10,
    "10 secs": DataFrequency.SECOND_10,
    "30s": DataFrequency.SECOND_30,
    "30 sec": DataFrequency.SECOND_30,
    "30 secs": DataFrequency.SECOND_30,
    "1m": DataFrequency.MINUTE_1,
    "1 min": DataFrequency.MINUTE_1,
    "1 mins": DataFrequency.MINUTE_1,
    "1 minute": DataFrequency.MINUTE_1,
    "1 minutes": DataFrequency.MINUTE_1,
    "5m": DataFrequency.MINUTE_5,
    "5 min": DataFrequency.MINUTE_5,
    "5 mins": DataFrequency.MINUTE_5,
    "5 minute": DataFrequency.MINUTE_5,
    "5 minutes": DataFrequency.MINUTE_5,
    "15m": DataFrequency.MINUTE_15,
    "15 min": DataFrequency.MINUTE_15,
    "15 mins": DataFrequency.MINUTE_15,
    "1h": DataFrequency.HOUR_1,
    "1 hour": DataFrequency.HOUR_1,
    "1 hours": DataFrequency.HOUR_1,
    "1d": DataFrequency.DAY_1,
    "1 day": DataFrequency.DAY_1,
    "1 days": DataFrequency.DAY_1,
    "daily": DataFrequency.DAY_1,
    "irregular": DataFrequency.IRREGULAR,
}

_BAR_SIZE_BY_FREQUENCY: dict[DataFrequency, str] = {
    DataFrequency.TICK: "1 secs",
    DataFrequency.SECOND_1: "1 secs",
    DataFrequency.SECOND_5: "5 secs",
    DataFrequency.SECOND_10: "10 secs",
    DataFrequency.SECOND_30: "30 secs",
    DataFrequency.MINUTE_1: "1 min",
    DataFrequency.MINUTE_5: "5 mins",
    DataFrequency.MINUTE_15: "15 mins",
    DataFrequency.HOUR_1: "1 hour",
    DataFrequency.DAY_1: "1 day",
    DataFrequency.IRREGULAR: "1 min",
}

_MARKET_PROVIDER_ALIASES: dict[str, MarketDataProvider] = {
    "file": MarketDataProvider.FILE,
    "files": MarketDataProvider.FILE,
    "csv": MarketDataProvider.FILE,
    "local": MarketDataProvider.FILE,
    "ib": MarketDataProvider.IB,
    "ibkr": MarketDataProvider.IB,
    "interactive_brokers": MarketDataProvider.IB,
    "interactive-brokers": MarketDataProvider.IB,
    "interactive brokers": MarketDataProvider.IB,
    "alpaca": MarketDataProvider.ALPACA,
}

_NEWS_PROVIDER_ALIASES: dict[str, NewsDataProvider] = {
    "": NewsDataProvider.NONE,
    "none": NewsDataProvider.NONE,
    "disabled": NewsDataProvider.NONE,
    "off": NewsDataProvider.NONE,
    "mock": NewsDataProvider.MOCK,
    "mock_news": NewsDataProvider.MOCK,
    "benzinga": NewsDataProvider.BENZINGA,
    "benzinga_news": NewsDataProvider.BENZINGA,
    "alphavantage": NewsDataProvider.ALPHAVANTAGE,
    "alpha_vantage": NewsDataProvider.ALPHAVANTAGE,
    "alpha-vantage": NewsDataProvider.ALPHAVANTAGE,
}

_INSTRUMENT_ALIASES: dict[str, InstrumentType] = {
    "stk": InstrumentType.STOCK,
    "stock": InstrumentType.STOCK,
    "equity": InstrumentType.STOCK,
    "opt": InstrumentType.OPTION,
    "option": InstrumentType.OPTION,
    "fut": InstrumentType.FUTURE,
    "future": InstrumentType.FUTURE,
    "crypto": InstrumentType.CRYPTO,
    "cryptocurrency": InstrumentType.CRYPTO,
    "ind": InstrumentType.INDEX,
    "index": InstrumentType.INDEX,
}


class MarketDataSourceConfig(BaseModel):
    """Market-data provider details for a canonical data request."""

    model_config = ConfigDict(extra="allow")

    provider: MarketDataProvider = MarketDataProvider.IB
    instrument_type: InstrumentType = InstrumentType.STOCK
    exchange: str = Field(default="SMART", min_length=1)
    currency: str = Field(default="USD", min_length=1)
    primary_exchange: str | None = None
    bar_size: str = Field(default="1 min", min_length=1)
    broker_data_type: str = Field(default="TRADES", min_length=1)
    explicit_files: dict[str, str] = Field(default_factory=dict)
    input_paths: list[str] = Field(default_factory=list)
    allow_empty_symbols: list[str] = Field(default_factory=list)

    @field_validator("provider", mode="before")
    @classmethod
    def _validate_provider(cls, value: object) -> MarketDataProvider:
        """Normalize provider aliases before enum validation."""

        return normalize_market_provider(value)

    @field_validator("instrument_type", mode="before")
    @classmethod
    def _validate_instrument_type(cls, value: object) -> InstrumentType:
        """Normalize instrument aliases before enum validation."""

        return normalize_instrument_type(value)

    @field_validator("exchange", "currency", "bar_size", "broker_data_type")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        """Trim required string fields."""

        normalized = value.strip()
        if not normalized:
            raise ValueError("value must be non-empty")
        return normalized

    @field_validator("primary_exchange", mode="before")
    @classmethod
    def _normalize_optional_text(cls, value: object) -> str | None:
        """Normalize blank optional strings to None."""

        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @field_validator("explicit_files", mode="before")
    @classmethod
    def _normalize_explicit_files(cls, value: object) -> dict[str, str]:
        """Normalize explicit file mappings by uppercasing symbol keys."""

        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("explicit_files must be a mapping of symbol to path")
        normalized: dict[str, str] = {}
        for raw_symbol, raw_path in value.items():
            symbol = str(raw_symbol).strip().upper()
            path = str(raw_path).strip()
            if not symbol or not path:
                continue
            normalized[symbol] = path
        return normalized

    @field_validator("input_paths", mode="before")
    @classmethod
    def _normalize_input_paths(cls, value: object) -> list[str]:
        """Normalize file-backed input paths to strings."""

        return _coerce_string_list(value)

    @field_validator("allow_empty_symbols", mode="before")
    @classmethod
    def _normalize_allow_empty_symbols(cls, value: object) -> list[str]:
        """Normalize explicitly allowed empty-data symbols."""

        values = _coerce_string_list(value)
        if not values:
            return []
        return normalize_symbols(values)

    @classmethod
    def from_config(
        cls,
        config: RawConfig,
        *,
        frequency: DataFrequency,
        symbols: Sequence[str],
    ) -> MarketDataSourceConfig:
        """Build market-source settings from loose model/job payloads."""

        payload = _copy_mapping(config)
        nested = _nested_mapping(
            payload,
            "market",
            "market_source",
            "market_data",
            "market_data_source",
        )

        explicit_files = _extract_symbol_file_mapping(payload, nested, symbols)
        input_paths = _extract_input_paths(payload, nested)
        raw_provider = _first_present(
            nested,
            "provider",
            "source",
            "market_provider",
            "data_provider",
        )
        if raw_provider is None:
            raw_provider = _first_present(
                payload,
                "market_provider",
                "broker_provider",
                "broker",
                "data_provider",
                "data_source",
            )
        if raw_provider is None:
            generic_provider = _first_present(payload, "provider", "source")
            if _is_market_provider(generic_provider):
                raw_provider = generic_provider
        if raw_provider is None and (explicit_files or input_paths):
            raw_provider = MarketDataProvider.FILE.value

        return cls(
            provider=raw_provider or MarketDataProvider.IB.value,
            instrument_type=_first_present(
                nested,
                "instrument_type",
                "secType",
                default=_first_present(payload, "instrument_type", "secType"),
            ),
            exchange=_first_present(
                nested,
                "exchange",
                default=_first_present(payload, "exchange", default="SMART"),
            ),
            currency=_first_present(
                nested,
                "currency",
                default=_first_present(payload, "currency", default="USD"),
            ),
            primary_exchange=_first_present(
                nested,
                "primary_exchange",
                "primaryExchange",
                default=_first_present(payload, "primary_exchange", "primaryExchange"),
            ),
            bar_size=_first_present(
                nested,
                "bar_size",
                "barSizeSetting",
                "historical_bar_size",
                default=_first_present(
                    payload,
                    "bar_size",
                    "barSizeSetting",
                    "historical_bar_size",
                    default=bar_size_for_frequency(frequency),
                ),
            ),
            broker_data_type=_first_present(
                nested,
                "broker_data_type",
                "whatToShow",
                default=_first_present(
                    payload,
                    "broker_data_type",
                    "whatToShow",
                    default="TRADES",
                ),
            ),
            explicit_files=explicit_files,
            input_paths=input_paths,
            allow_empty_symbols=_first_present(
                nested,
                "allow_empty_symbols",
                default=_first_present(payload, "allow_empty_symbols", default=[]),
            ),
        )


class NewsDataSourceConfig(BaseModel):
    """Historical news provider details for a canonical data request."""

    model_config = ConfigDict(extra="allow")

    provider: NewsDataProvider = NewsDataProvider.NONE
    enabled: bool = False
    required: bool = False
    include_body: bool = False
    limit: int = Field(default=100, ge=1)
    categories: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    fallback_provider: NewsDataProvider | None = None

    @field_validator("provider", mode="before")
    @classmethod
    def _validate_provider(cls, value: object) -> NewsDataProvider:
        """Normalize provider aliases before enum validation."""

        return normalize_news_provider(value)

    @field_validator("fallback_provider", mode="before")
    @classmethod
    def _validate_fallback_provider(cls, value: object) -> NewsDataProvider | None:
        """Normalize optional fallback provider aliases."""

        if value is None:
            return None
        provider = normalize_news_provider(value)
        return None if provider == NewsDataProvider.NONE else provider

    @field_validator("categories", "keywords", mode="before")
    @classmethod
    def _normalize_text_list(cls, value: object) -> list[str]:
        """Normalize list-like text filters."""

        return _coerce_string_list(value)

    @model_validator(mode="after")
    def _activate_required_or_selected_provider(self) -> NewsDataSourceConfig:
        """Keep enabled/required/provider fields internally consistent."""

        if self.required or self.provider != NewsDataProvider.NONE:
            self.enabled = True
        if not self.enabled:
            self.provider = NewsDataProvider.NONE
            self.required = False
        return self

    @classmethod
    def from_config(cls, config: RawConfig) -> NewsDataSourceConfig:
        """Build news-source settings from loose model/job payloads."""

        payload = _copy_mapping(config)
        raw_nested = _first_present(payload, "news", "news_source", "news_data")
        if isinstance(raw_nested, bool):
            nested: dict[str, object] = {"enabled": raw_nested}
        elif isinstance(raw_nested, Mapping):
            nested = {str(key): value for key, value in raw_nested.items()}
        else:
            nested = {}

        enabled = _first_present(
            nested,
            "enabled",
            default=_first_present(payload, "news_enabled", default=False),
        )
        required = _first_present(
            nested,
            "required",
            default=_first_present(payload, "news_required", default=False),
        )
        raw_provider = _first_present(
            nested,
            "provider",
            "source",
            default=_first_present(payload, "news_provider", "news_source"),
        )
        if raw_provider is None and (bool(enabled) or bool(required)):
            raw_provider = NewsDataProvider.BENZINGA.value

        return cls(
            provider=raw_provider or NewsDataProvider.NONE.value,
            enabled=_coerce_bool(enabled),
            required=_coerce_bool(required),
            include_body=_coerce_bool(
                _first_present(
                    nested,
                    "include_body",
                    "includeBody",
                    default=_first_present(payload, "include_news_body", default=False),
                )
            ),
            limit=_first_present(
                nested,
                "limit",
                "result_limit",
                default=_first_present(payload, "news_limit", default=100),
            ),
            categories=_first_present(
                nested,
                "categories",
                default=_first_present(payload, "news_categories", default=[]),
            ),
            keywords=_first_present(
                nested,
                "keywords",
                default=_first_present(payload, "news_keywords", default=[]),
            ),
            fallback_provider=_first_present(
                nested,
                "fallback_provider",
                "fallback",
                default=_first_present(payload, "news_fallback_provider"),
            ),
        )


class TrainingDataRequest(BaseModel):
    """Canonical merged data request used by training and backtesting."""

    symbols: list[str] = Field(min_length=1)
    start_time: datetime
    end_time: datetime
    frequency: DataFrequency
    market_source: MarketDataSourceConfig
    news_source: NewsDataSourceConfig = Field(default_factory=NewsDataSourceConfig)
    cache_policy: CachePolicy = CachePolicy.PREFER_CACHE

    @field_validator("symbols", mode="before")
    @classmethod
    def _normalize_symbols(cls, value: object) -> list[str]:
        """Normalize symbols to uppercase unique values."""

        return normalize_symbols(value)

    @field_validator("start_time", mode="before")
    @classmethod
    def _normalize_start_time(cls, value: object) -> datetime:
        """Normalize request start time to UTC."""

        return coerce_utc_datetime(value, boundary="start")

    @field_validator("end_time", mode="before")
    @classmethod
    def _normalize_end_time(cls, value: object) -> datetime:
        """Normalize request end time to UTC."""

        return coerce_utc_datetime(value, boundary="end")

    @field_validator("frequency", mode="before")
    @classmethod
    def _normalize_frequency(cls, value: object) -> DataFrequency:
        """Normalize loose UI frequency values to DataFrequency."""

        return normalize_frequency(value)

    @field_validator("cache_policy", mode="before")
    @classmethod
    def _normalize_cache_policy(cls, value: object) -> CachePolicy:
        """Normalize cache-policy aliases."""

        return normalize_cache_policy(value)

    @model_validator(mode="after")
    def _validate_range(self) -> TrainingDataRequest:
        """Validate chronological request bounds."""

        if self.end_time < self.start_time:
            raise ValueError("end_date must be on or after start_date")
        return self

    @classmethod
    def from_configs(
        cls,
        training_data_config: RawConfig,
        data_config: RawConfig = None,
    ) -> TrainingDataRequest:
        """Merge model training config with job/request overrides."""

        merged = merge_data_configs(training_data_config, data_config)
        return cls.from_config(merged)

    @classmethod
    def from_backtest_config(
        cls,
        model_data_config: RawConfig,
        *,
        start_date: object,
        end_date: object,
        symbols: object | None = None,
        overrides: RawConfig = None,
    ) -> TrainingDataRequest:
        """Build a data request for a backtest while preserving provider config."""

        merged = merge_data_configs(model_data_config, overrides)
        for key in (
            "start_time",
            "start_datetime",
            "start_date",
            "from",
            "date_from",
            "end_time",
            "end_datetime",
            "end_date",
            "to",
            "date_to",
        ):
            merged.pop(key, None)
        merged["start_date"] = start_date
        merged["end_date"] = end_date
        if symbols is not None:
            merged["symbols"] = symbols
        return cls.from_config(merged)

    @classmethod
    def from_config(cls, config: RawConfig) -> TrainingDataRequest:
        """Build a canonical request from one loose config mapping."""

        payload = _copy_mapping(config)
        symbols = normalize_symbols(
            _first_present(payload, "symbols", "symbol", "tickers", "ticker")
        )
        frequency = normalize_frequency(
            _first_present(
                payload,
                "frequency",
                "data_frequency",
                "input_frequency",
                "bar_size",
                "barSizeSetting",
            )
        )
        return cls(
            symbols=symbols,
            start_time=_first_present(
                payload,
                "start_time",
                "start_datetime",
                "start_date",
                "from",
                "date_from",
            ),
            end_time=_first_present(
                payload,
                "end_time",
                "end_datetime",
                "end_date",
                "to",
                "date_to",
            ),
            frequency=frequency,
            market_source=MarketDataSourceConfig.from_config(
                payload,
                frequency=frequency,
                symbols=symbols,
            ),
            news_source=NewsDataSourceConfig.from_config(payload),
            cache_policy=_first_present(
                payload, "cache_policy", default="prefer_cache"
            ),
        )

    def to_news_query(self, symbols: Sequence[str] | None = None) -> NewsQuery:
        """Convert the request to a provider-agnostic NewsQuery."""

        requested_symbols = list(symbols) if symbols is not None else list(self.symbols)
        normalized_symbols = normalize_symbols(requested_symbols)
        return NewsQuery(
            start_time=self.start_time,
            end_time=self.end_time,
            symbols=normalized_symbols,
            keywords=list(self.news_source.keywords),
            categories=list(self.news_source.categories),
            limit=self.news_source.limit,
            include_body=self.news_source.include_body,
        )


def merge_data_configs(
    base_config: RawConfig,
    override_config: RawConfig = None,
) -> dict[str, object]:
    """Deep-merge two loose data-source config dictionaries."""

    base = _copy_mapping(base_config)
    overrides = _copy_mapping(override_config)
    return _deep_merge(base, overrides)


def normalize_training_data_request(
    training_data_config: RawConfig,
    data_config: RawConfig = None,
) -> TrainingDataRequest:
    """Return the canonical request for a training job."""

    return TrainingDataRequest.from_configs(training_data_config, data_config)


def normalize_backtest_data_request(
    model_data_config: RawConfig,
    *,
    start_date: object,
    end_date: object,
    symbols: object | None = None,
    overrides: RawConfig = None,
) -> TrainingDataRequest:
    """Return the canonical request for a backtest job."""

    return TrainingDataRequest.from_backtest_config(
        model_data_config,
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        overrides=overrides,
    )


def validate_data_config_payload(
    payload: RawConfig,
    *,
    require_complete: bool = False,
) -> dict[str, object]:
    """Validate a loose data config without forcing legacy callers to be complete."""

    copied = _copy_mapping(payload)
    if not copied:
        if require_complete:
            raise DataSourceConfigError("data_config must include symbols and dates")
        return copied

    if require_complete:
        TrainingDataRequest.from_config(copied)
        return copied

    if _first_present(copied, "symbols", "symbol", "tickers", "ticker") is not None:
        normalize_symbols(
            _first_present(copied, "symbols", "symbol", "tickers", "ticker")
        )

    start_value = _first_present(
        copied, "start_time", "start_datetime", "start_date", "from", "date_from"
    )
    end_value = _first_present(
        copied, "end_time", "end_datetime", "end_date", "to", "date_to"
    )
    if start_value is not None:
        start_time = coerce_utc_datetime(start_value, boundary="start")
    else:
        start_time = None
    if end_value is not None:
        end_time = coerce_utc_datetime(end_value, boundary="end")
    else:
        end_time = None
    if start_time is not None and end_time is not None and end_time < start_time:
        raise DataSourceConfigError("end_date must be on or after start_date")

    frequency_value = _first_present(
        copied,
        "frequency",
        "data_frequency",
        "input_frequency",
        "bar_size",
        "barSizeSetting",
    )
    if frequency_value is not None:
        normalize_frequency(frequency_value)

    cache_policy_value = _first_present(copied, "cache_policy")
    if cache_policy_value is not None:
        normalize_cache_policy(cache_policy_value)

    _validate_optional_market_provider(copied)
    _validate_optional_news_provider(copied)
    return copied


def normalize_symbols(value: object) -> list[str]:
    """Normalize a loose symbol or symbol collection to uppercase values."""

    if value is None:
        raise DataSourceConfigError("data request must include at least one symbol")
    if isinstance(value, str):
        candidates = value.replace(";", ",").split(",")
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        candidates = [str(item) for item in value]
    else:
        raise DataSourceConfigError("symbols must be a string or list of strings")

    normalized: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        symbol = str(candidate).strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        normalized.append(symbol)
    if not normalized:
        raise DataSourceConfigError("data request must include at least one symbol")
    return normalized


def normalize_frequency(value: object) -> DataFrequency:
    """Normalize UI and broker frequency aliases to DataFrequency."""

    if isinstance(value, DataFrequency):
        return value
    if value is None:
        raise DataSourceConfigError("data_frequency is required")

    text = str(value).strip()
    if not text:
        raise DataSourceConfigError("data_frequency must be non-empty")

    normalized = text.lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    alias = _FREQUENCY_ALIASES.get(normalized)
    if alias is not None:
        return alias

    compact_alias = _FREQUENCY_ALIASES.get(normalized.replace(" ", ""))
    if compact_alias is not None:
        return compact_alias

    try:
        return DataFrequency(text)
    except ValueError:
        pass

    member_name = text.strip().upper()
    if member_name in DataFrequency.__members__:
        return DataFrequency[member_name]

    supported = ", ".join(
        sorted({*list(_FREQUENCY_ALIASES), *DataFrequency.__members__})
    )
    raise DataSourceConfigError(
        f"Unsupported data_frequency '{text}'. Supported values include: {supported}"
    )


def normalize_cache_policy(value: object) -> CachePolicy:
    """Normalize cache-policy strings to CachePolicy."""

    if isinstance(value, CachePolicy):
        return value
    text = (
        str(value or CachePolicy.PREFER_CACHE.value).strip().lower().replace("-", "_")
    )
    aliases = {
        "cache": CachePolicy.PREFER_CACHE,
        "prefer": CachePolicy.PREFER_CACHE,
        "prefer_cache": CachePolicy.PREFER_CACHE,
        "refresh": CachePolicy.REFRESH,
        "reload": CachePolicy.REFRESH,
        "force_refresh": CachePolicy.REFRESH,
        "require": CachePolicy.REQUIRE_CACHE,
        "require_cache": CachePolicy.REQUIRE_CACHE,
        "cache_only": CachePolicy.REQUIRE_CACHE,
    }
    policy = aliases.get(text)
    if policy is None:
        raise DataSourceConfigError(
            "cache_policy must be one of prefer_cache, refresh, or require_cache"
        )
    return policy


def normalize_market_provider(value: object) -> MarketDataProvider:
    """Normalize market provider aliases to MarketDataProvider."""

    if isinstance(value, MarketDataProvider):
        return value
    text = str(value or MarketDataProvider.IB.value).strip().lower()
    provider = _MARKET_PROVIDER_ALIASES.get(text)
    if provider is None:
        raise DataSourceConfigError(
            "market provider must be one of file, ib, or alpaca"
        )
    return provider


def normalize_news_provider(value: object) -> NewsDataProvider:
    """Normalize news provider aliases to NewsDataProvider."""

    if isinstance(value, NewsDataProvider):
        return value
    text = str(value or NewsDataProvider.NONE.value).strip().lower()
    provider = _NEWS_PROVIDER_ALIASES.get(text)
    if provider is None:
        raise DataSourceConfigError(
            "news provider must be one of none, mock, benzinga, or alphavantage"
        )
    return provider


def normalize_instrument_type(value: object) -> InstrumentType:
    """Normalize instrument aliases to InstrumentType."""

    if isinstance(value, InstrumentType):
        return value
    text = str(value or InstrumentType.STOCK.value).strip().lower()
    instrument_type = _INSTRUMENT_ALIASES.get(text)
    if instrument_type is None:
        try:
            return InstrumentType(str(value).strip().upper())
        except ValueError as exc:
            raise DataSourceConfigError(
                f"Unsupported instrument_type '{value}'"
            ) from exc
    return instrument_type


def coerce_utc_datetime(value: object, *, boundary: str) -> datetime:
    """Coerce dates or datetimes to timezone-aware UTC datetimes."""

    if value is None:
        raise DataSourceConfigError(f"{boundary}_date is required")
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(
            value,
            time.min if boundary == "start" else time.max,
            tzinfo=UTC,
        )
    else:
        parsed = _parse_datetime_text(str(value), boundary=boundary)

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def bar_size_for_frequency(frequency: DataFrequency) -> str:
    """Return the broker bar-size string matching a DataFrequency."""

    return _BAR_SIZE_BY_FREQUENCY[frequency]


def _copy_mapping(config: RawConfig) -> dict[str, object]:
    if config is None:
        return {}
    if not isinstance(config, Mapping):
        raise DataSourceConfigError("data_config must be a dictionary")
    return {str(key): value for key, value in config.items()}


def _deep_merge(
    base: dict[str, object],
    overrides: dict[str, object],
) -> dict[str, object]:
    merged = dict(base)
    for key, value in overrides.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(
                {
                    str(nested_key): nested_value
                    for nested_key, nested_value in existing.items()
                },
                {
                    str(nested_key): nested_value
                    for nested_key, nested_value in value.items()
                },
            )
        else:
            merged[key] = value
    return merged


def _first_present(
    payload: Mapping[str, object],
    *keys: str,
    default: object | None = None,
) -> object | None:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return default


def _nested_mapping(payload: Mapping[str, object], *keys: str) -> dict[str, object]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, Mapping):
            return {
                str(nested_key): nested_value
                for nested_key, nested_value in value.items()
            }
    return {}


def _extract_symbol_file_mapping(
    payload: Mapping[str, object],
    nested: Mapping[str, object],
    symbols: Sequence[str],
) -> dict[str, str]:
    raw_files = _first_present(
        nested,
        "explicit_files",
        "symbol_files",
        "data_files",
        default=_first_present(payload, "explicit_files", "symbol_files", "data_files"),
    )
    if isinstance(raw_files, Mapping):
        return {
            str(symbol).strip().upper(): str(path).strip()
            for symbol, path in raw_files.items()
            if str(symbol).strip() and str(path).strip()
        }
    if isinstance(raw_files, (str, Path)) and len(symbols) == 1:
        return {str(symbols[0]).upper(): str(raw_files)}
    return {}


def _extract_input_paths(
    payload: Mapping[str, object],
    nested: Mapping[str, object],
) -> list[str]:
    values: list[str] = []
    for raw_value in (
        _first_present(nested, "input_paths", "file_paths"),
        _first_present(payload, "input_paths", "file_paths"),
        _first_present(
            payload,
            "market_data_path",
            "backtest_data_path",
            "csv_path",
            "file_path",
            "data_path",
        ),
    ):
        values.extend(_coerce_string_list(raw_value))
    return list(dict.fromkeys(values))


def _coerce_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        normalized = str(value).strip()
        return [normalized] if normalized else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _is_market_provider(value: object) -> bool:
    if value is None:
        return False
    try:
        normalize_market_provider(value)
    except DataSourceConfigError:
        return False
    return True


def _validate_optional_market_provider(payload: Mapping[str, object]) -> None:
    nested = _nested_mapping(
        payload,
        "market",
        "market_source",
        "market_data",
        "market_data_source",
    )
    raw_provider = _first_present(
        nested,
        "provider",
        "source",
        default=_first_present(
            payload,
            "market_provider",
            "broker_provider",
            "broker",
            "data_provider",
            "data_source",
        ),
    )
    if raw_provider is not None:
        normalize_market_provider(raw_provider)


def _validate_optional_news_provider(payload: Mapping[str, object]) -> None:
    raw_nested = _first_present(payload, "news", "news_source", "news_data")
    nested = raw_nested if isinstance(raw_nested, Mapping) else {}
    raw_provider = _first_present(
        nested,
        "provider",
        "source",
        default=_first_present(payload, "news_provider", "news_source"),
    )
    if raw_provider is not None:
        normalize_news_provider(raw_provider)


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    raise DataSourceConfigError(f"Cannot interpret boolean value '{value}'")


def _parse_datetime_text(value: str, *, boundary: str) -> datetime:
    normalized = value.strip()
    if not normalized:
        raise DataSourceConfigError(f"{boundary}_date must be non-empty")

    try:
        parsed_date = date.fromisoformat(normalized)
    except ValueError:
        parsed_date = None
    if parsed_date is not None:
        return datetime.combine(
            parsed_date,
            time.min if boundary == "start" else time.max,
            tzinfo=UTC,
        )

    datetime_text = normalized.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(datetime_text)
    except ValueError as exc:
        raise DataSourceConfigError(
            f"{boundary}_date must be an ISO date or datetime"
        ) from exc


__all__ = [
    "CachePolicy",
    "DataSourceConfigError",
    "MarketDataProvider",
    "MarketDataSourceConfig",
    "NewsDataProvider",
    "NewsDataSourceConfig",
    "TrainingDataRequest",
    "bar_size_for_frequency",
    "coerce_utc_datetime",
    "merge_data_configs",
    "normalize_backtest_data_request",
    "normalize_cache_policy",
    "normalize_frequency",
    "normalize_market_provider",
    "normalize_news_provider",
    "normalize_symbols",
    "normalize_training_data_request",
    "validate_data_config_payload",
]
