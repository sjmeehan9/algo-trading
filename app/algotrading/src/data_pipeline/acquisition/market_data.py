"""Historical market-data acquisition backed by cache, files, or brokers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path

import pandas as pd
from algotrading.api.schemas.data_sources import (
    CachePolicy,
    MarketDataProvider,
    TrainingDataRequest,
)
from algotrading.src.broker import BrokerAdapter, BrokerRegistry
from algotrading.src.data_pipeline import DataBatch, DataFrequency, DataRecord, DataType
from algotrading.src.data_pipeline.sources.broker_source import (
    BrokerDataSource,
    ContractDefaults,
)
from algotrading.src.data_pipeline.sources.exceptions import (
    DataSourceError,
    DataValidationError,
)
from algotrading.src.data_pipeline.storage import (
    DataManifest,
    LocalDataStore,
    LocalDataStoreError,
)
from pydantic import BaseModel, ConfigDict, Field


class AcquisitionStatus(str, Enum):
    """Per-symbol market-data acquisition status values."""

    CACHED = "cached"
    ACQUIRED = "acquired"
    EMPTY_ALLOWED = "empty_allowed"
    OPTIONAL_UNAVAILABLE = "optional_unavailable"


class SymbolAcquisitionReport(BaseModel):
    """Acquisition details for one requested market-data symbol."""

    model_config = ConfigDict(use_enum_values=True)

    symbol: str
    status: AcquisitionStatus
    provider: str
    row_count: int = Field(ge=0)
    data_path: str | None = None
    manifest_path: str | None = None
    actual_start: datetime | None = None
    actual_end: datetime | None = None
    source_file_paths: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AcquisitionResult(BaseModel):
    """Historical market-data acquisition result for a full request."""

    model_config = ConfigDict(use_enum_values=True)

    provider: str
    cache_policy: str
    frequency: DataFrequency
    requested_start: datetime
    requested_end: datetime
    reports: list[SymbolAcquisitionReport]
    warnings: list[str] = Field(default_factory=list)

    @property
    def symbols(self) -> list[str]:
        """Return symbols covered by this result."""

        return [report.symbol for report in self.reports]


class MarketDataAcquisitionError(DataSourceError):
    """Raised when historical market data cannot be acquired."""

    def __init__(
        self,
        message: str,
        reports: Sequence[SymbolAcquisitionReport] | None = None,
    ) -> None:
        """Initialize an acquisition error with optional partial reports."""

        super().__init__(message)
        self.reports = list(reports or [])


class HistoricalMarketDataAcquirer:
    """Ensure historical market bars exist in the canonical local store."""

    def __init__(
        self,
        *,
        store: LocalDataStore | None = None,
        broker_registry: BrokerRegistry | None = None,
        broker_config: Mapping[str, object] | None = None,
        broker_connection_params: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize the acquirer with storage and broker collaborators."""

        self._store = store or LocalDataStore()
        self._broker_registry = broker_registry or BrokerRegistry()
        self._broker_config = dict(broker_config) if broker_config is not None else None
        self._broker_connection_params = (
            dict(broker_connection_params)
            if broker_connection_params is not None
            else None
        )

    @property
    def store(self) -> LocalDataStore:
        """Return the canonical local data store used by this acquirer."""

        return self._store

    def acquire(self, request: TrainingDataRequest) -> AcquisitionResult:
        """Acquire or load historical market data for a canonical request."""

        warnings: list[str] = []
        if request.cache_policy != CachePolicy.REFRESH:
            try:
                cached = self._store.find_market_data(request)
            except LocalDataStoreError as exc:
                if request.cache_policy == CachePolicy.REQUIRE_CACHE:
                    raise MarketDataAcquisitionError(
                        f"Required market-data cache is unavailable: {exc}"
                    ) from exc
                warnings.append(f"Market-data cache miss: {exc}")
            else:
                reports = [
                    _report_from_manifest(
                        symbol=symbol,
                        manifest=stored.manifest,
                        status=AcquisitionStatus.CACHED,
                        row_count=len(stored.batch.records),
                        data_path=stored.data_path,
                        manifest_path=stored.manifest_path,
                    )
                    for symbol, stored in sorted(cached.items())
                ]
                return _result_from_reports(request, reports, warnings=[])

        provider = request.market_source.provider
        if provider == MarketDataProvider.FILE:
            reports = self._acquire_from_files(request)
        else:
            reports = self._acquire_from_broker(request)
        return _result_from_reports(request, reports, warnings=warnings)

    def _acquire_from_files(
        self,
        request: TrainingDataRequest,
    ) -> list[SymbolAcquisitionReport]:
        reports: list[SymbolAcquisitionReport] = []
        for symbol in request.symbols:
            source_path = _resolve_file_for_symbol(request, symbol)
            batch = _load_file_batch(
                path=source_path,
                symbol=symbol,
                start=request.start_time,
                end=request.end_time,
                frequency=request.frequency,
            )
            if not batch.records:
                reports.append(_empty_allowed_report(request, symbol))
                continue

            manifest = self._store.write_market_batch(
                batch,
                provider=request.market_source.provider,
                frequency=request.frequency,
                requested_start=request.start_time,
                requested_end=request.end_time,
                cache_policy=request.cache_policy,
                source_file_paths=[source_path],
            )
            reports.append(
                _report_from_manifest(
                    symbol=symbol,
                    manifest=manifest,
                    status=AcquisitionStatus.ACQUIRED,
                    row_count=len(batch.records),
                )
            )
        return reports

    def _acquire_from_broker(
        self,
        request: TrainingDataRequest,
    ) -> list[SymbolAcquisitionReport]:
        provider_name = _broker_registry_name(request.market_source.provider)
        adapter = self._create_broker_adapter(provider_name)
        source = BrokerDataSource(
            adapter=adapter,
            historical_bar_size=request.market_source.bar_size,
            broker_data_type=request.market_source.broker_data_type,
            historical_frequency=request.frequency,
            contract_defaults=ContractDefaults(
                instrument_type=request.market_source.instrument_type,
                exchange=request.market_source.exchange,
                currency=request.market_source.currency,
                primary_exchange=request.market_source.primary_exchange,
            ),
        )

        reports: list[SymbolAcquisitionReport] = []
        try:
            source.connect()
            for symbol in request.symbols:
                batch = _fetch_chunked_batch(source, request, symbol)
                if not batch.records:
                    reports.append(_empty_allowed_report(request, symbol))
                    continue

                manifest = self._store.write_market_batch(
                    batch,
                    provider=request.market_source.provider,
                    frequency=request.frequency,
                    requested_start=request.start_time,
                    requested_end=request.end_time,
                    cache_policy=request.cache_policy,
                )
                reports.append(
                    _report_from_manifest(
                        symbol=symbol,
                        manifest=manifest,
                        status=AcquisitionStatus.ACQUIRED,
                        row_count=len(batch.records),
                    )
                )
        finally:
            source.disconnect()
        return reports

    def _create_broker_adapter(self, provider_name: str) -> BrokerAdapter:
        if self._broker_config is None and self._broker_connection_params is None:
            return self._broker_registry.create_broker(provider_name)
        return self._broker_registry.create_broker(
            provider_name,
            config=self._broker_config or {},
            connection_params=self._broker_connection_params or {},
        )


def _result_from_reports(
    request: TrainingDataRequest,
    reports: list[SymbolAcquisitionReport],
    warnings: list[str],
) -> AcquisitionResult:
    return AcquisitionResult(
        provider=_provider_value(request.market_source.provider),
        cache_policy=request.cache_policy.value,
        frequency=request.frequency,
        requested_start=request.start_time,
        requested_end=request.end_time,
        reports=reports,
        warnings=warnings,
    )


def _report_from_manifest(
    *,
    symbol: str,
    manifest: DataManifest,
    status: AcquisitionStatus,
    row_count: int,
    data_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> SymbolAcquisitionReport:
    data_paths = list(manifest.data_file_paths)
    source_paths = list(manifest.source_file_paths)
    resolved_data_path = str(data_path or (data_paths[0] if data_paths else "")) or None
    resolved_manifest_path = str(manifest_path or manifest.manifest_path or "") or None
    return SymbolAcquisitionReport(
        symbol=symbol,
        status=status,
        provider=manifest.provider,
        row_count=row_count,
        data_path=resolved_data_path,
        manifest_path=resolved_manifest_path,
        actual_start=manifest.actual_start,
        actual_end=manifest.actual_end,
        source_file_paths=source_paths,
    )


def _empty_allowed_report(
    request: TrainingDataRequest,
    symbol: str,
) -> SymbolAcquisitionReport:
    if symbol not in set(request.market_source.allow_empty_symbols):
        provider = _provider_value(request.market_source.provider)
        raise MarketDataAcquisitionError(
            f"Provider '{provider}' returned no historical market bars for "
            f"symbol '{symbol}' between {request.start_time.isoformat()} and "
            f"{request.end_time.isoformat()}"
        )
    return SymbolAcquisitionReport(
        symbol=symbol,
        status=AcquisitionStatus.EMPTY_ALLOWED,
        provider=_provider_value(request.market_source.provider),
        row_count=0,
        warnings=["No market bars were returned; symbol is explicitly allowed empty."],
    )


def _resolve_file_for_symbol(request: TrainingDataRequest, symbol: str) -> Path:
    explicit_path = request.market_source.explicit_files.get(symbol)
    if explicit_path:
        return _existing_file_path(explicit_path, symbol)

    candidates = list(_iter_input_paths(request.market_source.input_paths))
    symbol_matches = [
        candidate
        for candidate in candidates
        if symbol.upper() in candidate.stem.upper().replace("-", "_").split("_")
    ]
    if symbol_matches:
        return symbol_matches[0]
    if len(candidates) == 1 and len(request.symbols) == 1:
        return candidates[0]
    raise MarketDataAcquisitionError(
        f"No explicit market-data CSV path configured for symbol '{symbol}'"
    )


def _iter_input_paths(raw_paths: Sequence[str]) -> Iterable[Path]:
    for raw_path in raw_paths:
        path = Path(raw_path).expanduser()
        if path.is_dir():
            yield from sorted(path.glob("*.csv"))
        elif path.exists():
            yield path


def _existing_file_path(raw_path: str, symbol: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.exists() or not path.is_file():
        raise MarketDataAcquisitionError(
            f"Configured market-data CSV for symbol '{symbol}' does not exist: {path}"
        )
    return path


def _load_file_batch(
    *,
    path: Path,
    symbol: str,
    start: datetime,
    end: datetime,
    frequency: DataFrequency,
) -> DataBatch:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise MarketDataAcquisitionError(
            f"Failed to read market CSV {path}: {exc}"
        ) from exc

    columns = {column.strip().lower(): column for column in frame.columns}
    timestamp_column = _first_column(columns, "timestamp", "date", "datetime", "time")
    required = {
        "open": _first_column(columns, "open"),
        "high": _first_column(columns, "high"),
        "low": _first_column(columns, "low"),
        "close": _first_column(columns, "close"),
        "volume": _first_column(columns, "volume"),
    }
    missing = [
        name
        for name, column in {"timestamp": timestamp_column, **required}.items()
        if column is None
    ]
    if missing:
        raise MarketDataAcquisitionError(
            f"Market CSV {path} is missing required columns: {', '.join(missing)}"
        )

    working = frame.copy()
    timestamps = pd.to_datetime(working[timestamp_column], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise MarketDataAcquisitionError(
            f"Market CSV {path} contains invalid timestamp values"
        )
    working["_timestamp"] = timestamps

    symbol_column = _first_column(columns, "symbol", "ticker")
    if symbol_column is not None:
        working = working[working[symbol_column].astype(str).str.upper() == symbol]
    start_utc = _ensure_utc(start)
    end_utc = _ensure_utc(end)
    padding = _bar_interval_for_frequency(frequency)
    working = working[
        (working["_timestamp"] >= pd.Timestamp(start_utc - padding))
        & (working["_timestamp"] <= pd.Timestamp(end_utc + padding))
    ].sort_values("_timestamp")

    records = [
        _record_from_file_row(
            row,
            symbol=symbol,
            source_id=f"file:{path.name}",
            frequency=frequency,
            required_columns=required,
            vwap_column=_first_column(columns, "vwap", "wap"),
            trade_count_column=_first_column(columns, "trade_count", "count"),
            source_id_column=_first_column(columns, "source_id"),
        )
        for row in working.to_dict("records")
    ]
    return DataBatch(
        records=records,
        start_time=records[0].timestamp if records else start_utc,
        end_time=records[-1].timestamp if records else end_utc,
        data_type=DataType.MARKET_BAR,
        symbol=symbol,
    )


def _record_from_file_row(
    row: Mapping[str, object],
    *,
    symbol: str,
    source_id: str,
    frequency: DataFrequency,
    required_columns: Mapping[str, str | None],
    vwap_column: str | None,
    trade_count_column: str | None,
    source_id_column: str | None,
) -> DataRecord:
    payload = {
        "open": _required_float(row, required_columns["open"], "open"),
        "high": _required_float(row, required_columns["high"], "high"),
        "low": _required_float(row, required_columns["low"], "low"),
        "close": _required_float(row, required_columns["close"], "close"),
        "volume": _required_int(row, required_columns["volume"], "volume"),
        "vwap": _optional_float(row.get(vwap_column) if vwap_column else None),
        "trade_count": _optional_int(
            row.get(trade_count_column) if trade_count_column else None
        ),
    }
    return DataRecord(
        timestamp=_ensure_utc(row["_timestamp"].to_pydatetime()),
        data_type=DataType.MARKET_BAR,
        symbol=symbol,
        payload=payload,
        source_id=str(row.get(source_id_column) or source_id),
        frequency=frequency,
    )


def _fetch_chunked_batch(
    source: BrokerDataSource,
    request: TrainingDataRequest,
    symbol: str,
) -> DataBatch:
    records_by_key: dict[tuple[str, datetime], DataRecord] = {}
    padding = _bar_interval_for_frequency(request.frequency)
    for chunk_start, chunk_end in _chunk_ranges(
        request.start_time,
        request.end_time,
        request.frequency,
    ):
        batch = source.fetch_batch(
            symbol=symbol,
            start=chunk_start - padding,
            end=chunk_end + padding,
            data_type=DataType.MARKET_BAR,
        )
        for record in batch.records:
            normalized = _normalize_record(record, symbol, request.frequency)
            records_by_key[(symbol, normalized.timestamp)] = normalized

    records = sorted(records_by_key.values(), key=lambda item: item.timestamp)
    return DataBatch(
        records=records,
        start_time=records[0].timestamp if records else request.start_time,
        end_time=records[-1].timestamp if records else request.end_time,
        data_type=DataType.MARKET_BAR,
        symbol=symbol,
    )


def _normalize_record(
    record: DataRecord,
    symbol: str,
    frequency: DataFrequency,
) -> DataRecord:
    return DataRecord(
        timestamp=_ensure_utc(record.timestamp),
        data_type=DataType.MARKET_BAR,
        symbol=symbol,
        payload=dict(record.payload),
        source_id=record.source_id,
        frequency=frequency,
    )


def _chunk_ranges(
    start: datetime,
    end: datetime,
    frequency: DataFrequency,
) -> Iterable[tuple[datetime, datetime]]:
    start_utc = _ensure_utc(start)
    end_utc = _ensure_utc(end)
    chunk_size = _chunk_size_for_frequency(frequency)
    current = start_utc
    while current < end_utc:
        chunk_end = min(end_utc, current + chunk_size)
        yield current, chunk_end
        if chunk_end >= end_utc:
            break
        current = chunk_end
    if start_utc == end_utc:
        yield start_utc, end_utc


def _chunk_size_for_frequency(frequency: DataFrequency) -> timedelta:
    if frequency in {
        DataFrequency.TICK,
        DataFrequency.SECOND_1,
        DataFrequency.SECOND_5,
        DataFrequency.SECOND_10,
        DataFrequency.SECOND_30,
        DataFrequency.MINUTE_1,
    }:
        return timedelta(days=7)
    if frequency in {DataFrequency.MINUTE_5, DataFrequency.MINUTE_15}:
        return timedelta(days=30)
    if frequency == DataFrequency.HOUR_1:
        return timedelta(days=365)
    return timedelta(days=3650)


def _bar_interval_for_frequency(frequency: DataFrequency) -> timedelta:
    if frequency == DataFrequency.TICK:
        return timedelta(seconds=1)
    if frequency == DataFrequency.SECOND_1:
        return timedelta(seconds=1)
    if frequency == DataFrequency.SECOND_5:
        return timedelta(seconds=5)
    if frequency == DataFrequency.SECOND_10:
        return timedelta(seconds=10)
    if frequency == DataFrequency.SECOND_30:
        return timedelta(seconds=30)
    if frequency == DataFrequency.MINUTE_1:
        return timedelta(minutes=1)
    if frequency == DataFrequency.MINUTE_5:
        return timedelta(minutes=5)
    if frequency == DataFrequency.MINUTE_15:
        return timedelta(minutes=15)
    if frequency == DataFrequency.HOUR_1:
        return timedelta(hours=1)
    if frequency == DataFrequency.DAY_1:
        return timedelta(days=1)
    return timedelta(minutes=1)


def _broker_registry_name(provider: MarketDataProvider) -> str:
    if provider == MarketDataProvider.IB:
        return "interactive_brokers"
    if provider == MarketDataProvider.ALPACA:
        return "alpaca"
    raise MarketDataAcquisitionError(
        f"Provider '{provider.value}' does not support broker-backed acquisition"
    )


def _provider_value(provider: object) -> str:
    return provider.value if isinstance(provider, Enum) else str(provider)


def _first_column(columns: Mapping[str, str], *candidates: str) -> str | None:
    for candidate in candidates:
        column = columns.get(candidate.lower())
        if column is not None:
            return column
    return None


def _required_float(
    row: Mapping[str, object],
    column: str | None,
    label: str,
) -> float:
    if column is None:
        raise DataValidationError(f"Required market column '{label}' is missing")
    return _coerce_float(row.get(column), label)


def _required_int(
    row: Mapping[str, object],
    column: str | None,
    label: str,
) -> int:
    if column is None:
        raise DataValidationError(f"Required market column '{label}' is missing")
    value = row.get(column)
    if _is_missing(value):
        raise DataValidationError(f"Required market column '{label}' is missing")
    try:
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            f"Required market column '{label}' must be an integer"
        ) from exc


def _optional_float(value: object) -> float | None:
    if _is_missing(value):
        return None
    return _coerce_float(value, "optional market value")


def _optional_int(value: object) -> int | None:
    if _is_missing(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise DataValidationError("optional market value must be an integer") from exc


def _coerce_float(value: object, label: str) -> float:
    if _is_missing(value):
        raise DataValidationError(f"Required market column '{label}' is missing")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            f"Required market column '{label}' must be numeric"
        ) from exc


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


__all__ = [
    "AcquisitionResult",
    "AcquisitionStatus",
    "HistoricalMarketDataAcquirer",
    "MarketDataAcquisitionError",
    "SymbolAcquisitionReport",
]
