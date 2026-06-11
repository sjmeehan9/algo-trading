"""Historical market-data acquisition backed by cache, files, or brokers."""

from __future__ import annotations

import logging
import time as time_module
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, time, timedelta
from enum import Enum
from pathlib import Path
from threading import Event
from zoneinfo import ZoneInfo

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


logger = logging.getLogger(__name__)


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


class MarketDataAcquisitionCancelled(MarketDataAcquisitionError):
    """Raised when historical market-data acquisition is cancelled mid-flight."""


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

    def acquire(
        self,
        request: TrainingDataRequest,
        *,
        cancel_event: Event | None = None,
    ) -> AcquisitionResult:
        """Acquire or load historical market data for a canonical request.

        Args:
            request: Canonical market-data request.
            cancel_event: Optional event signalling that broker acquisition
                should stop; checked between requests so a long broker source
                can be interrupted cleanly.
        """

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
            reports = self._acquire_from_broker(request, cancel_event=cancel_event)
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
        *,
        cancel_event: Event | None = None,
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

        # The chunk plan is identical for every symbol (it depends only on the
        # range, frequency, and session window), so compute it once to size the
        # shared pacer and report progress.
        chunks = _plan_chunks(request)
        pacer = _build_pacer(
            request=request,
            chunks_per_symbol=len(chunks),
            symbol_count=len(request.symbols),
        )

        reports: list[SymbolAcquisitionReport] = []
        try:
            source.connect()
            for symbol in request.symbols:
                batch = _fetch_chunked_batch(
                    source,
                    request,
                    symbol,
                    chunks=chunks,
                    pacer=pacer,
                    cancel_event=cancel_event,
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


# Interactive Brokers pacing: at most 60 historical requests in any rolling
# 10-minute window. A small safety margin avoids brushing the limit.
_PACING_WINDOW_SECONDS = 600.0
_MAX_REQUESTS_PER_WINDOW = 55
# Steady spacing that keeps a long job exactly at the window limit without ever
# front-loading a burst (which would otherwise force a multi-minute stall once
# the window fills). ~10.9s per request.
_STEADY_INTERVAL_SECONDS = _PACING_WINDOW_SECONDS / _MAX_REQUESTS_PER_WINDOW
# Granularity for interruptible sleeps so cancellation is observed promptly.
_SLEEP_SLICE_SECONDS = 0.25


class _RequestPacer:
    """Rate-limit broker requests within IB's rolling-window pacing limits.

    A single pacer is shared across every symbol in one acquisition so the
    rolling-window cap reflects IB's per-connection limit rather than resetting
    per symbol. ``min_spacing`` is raised to a steady interval for large jobs so
    requests are evenly spaced instead of bursting and then stalling for minutes.
    """

    def __init__(self, min_spacing: float) -> None:
        self._min_spacing = max(0.0, min_spacing)
        self._request_times: deque[float] = deque()

    def wait(self, cancel_event: "Event | None" = None) -> None:
        """Block until the next request may proceed, honoring cancellation."""

        now = time_module.monotonic()
        self._evict_expired(now)

        if len(self._request_times) >= _MAX_REQUESTS_PER_WINDOW:
            wait_for = _PACING_WINDOW_SECONDS - (now - self._request_times[0])
            self._interruptible_sleep(wait_for, cancel_event)
            now = time_module.monotonic()
            self._evict_expired(now)

        if self._min_spacing > 0 and self._request_times:
            gap = now - self._request_times[-1]
            self._interruptible_sleep(self._min_spacing - gap, cancel_event)
            now = time_module.monotonic()

        self._request_times.append(now)

    def _evict_expired(self, now: float) -> None:
        while (
            self._request_times
            and now - self._request_times[0] >= _PACING_WINDOW_SECONDS
        ):
            self._request_times.popleft()

    @staticmethod
    def _interruptible_sleep(seconds: float, cancel_event: "Event | None") -> None:
        remaining = seconds
        while remaining > 0:
            if cancel_event is not None and cancel_event.is_set():
                return
            time_module.sleep(min(_SLEEP_SLICE_SECONDS, remaining))
            remaining -= _SLEEP_SLICE_SECONDS


def _plan_chunks(request: TrainingDataRequest) -> list[tuple[datetime, datetime]]:
    """Return the broker request windows for a request (symbol-independent)."""

    padding = _bar_interval_for_frequency(request.frequency)
    session = request.market_source.session_window()

    # Each chunk is widened by ``padding`` on both sides below to capture
    # boundary bars, so shrink the request span by that allowance to keep the
    # resulting duration within the IB per-bar-size maximum.
    max_span = _max_request_span(request.frequency) - 2 * padding
    if max_span <= timedelta(0):
        max_span = _max_request_span(request.frequency)

    return list(
        _chunk_ranges(
            request.start_time,
            request.end_time,
            max_span=max_span,
            session=session if _is_intraday_frequency(request.frequency) else None,
        )
    )


def _build_pacer(
    *,
    request: TrainingDataRequest,
    chunks_per_symbol: int,
    symbol_count: int,
) -> _RequestPacer:
    """Construct the shared pacer, smoothing spacing for large jobs.

    For jobs whose total request count exceeds the rolling-window cap, the
    minimum spacing is raised to a steady interval so requests are evenly paced
    rather than bursting and then stalling for minutes once the window fills.
    Small jobs keep the faster configured spacing.
    """

    configured_spacing = request.market_source.request_pacing_seconds
    total_requests = chunks_per_symbol * symbol_count
    if total_requests > _MAX_REQUESTS_PER_WINDOW:
        min_spacing = max(configured_spacing, _STEADY_INTERVAL_SECONDS)
    else:
        min_spacing = configured_spacing

    if total_requests > 0:
        estimated_minutes = (total_requests * min_spacing) / 60.0
        logger.info(
            "Sourcing historical market data | symbols=%s frequency=%s "
            "requests=%d spacing=%.1fs est_minutes=%.1f",
            ",".join(request.symbols),
            request.frequency.value,
            total_requests,
            min_spacing,
            estimated_minutes,
        )
    return _RequestPacer(min_spacing)


def _fetch_chunked_batch(
    source: BrokerDataSource,
    request: TrainingDataRequest,
    symbol: str,
    *,
    chunks: Sequence[tuple[datetime, datetime]],
    pacer: _RequestPacer,
    cancel_event: Event | None = None,
) -> DataBatch:
    records_by_key: dict[tuple[str, datetime], DataRecord] = {}
    padding = _bar_interval_for_frequency(request.frequency)
    total = len(chunks)

    for index, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        if cancel_event is not None and cancel_event.is_set():
            raise MarketDataAcquisitionCancelled(
                f"Historical market-data acquisition cancelled for '{symbol}' "
                f"after {index - 1}/{total} requests"
            )

        # Stagger successive broker requests to stay under IB pacing limits:
        # a minimum spacing avoids the 6-requests-per-2-seconds rule, and the
        # rolling window keeps within 60 historical requests per 10 minutes.
        pacer.wait(cancel_event)
        if cancel_event is not None and cancel_event.is_set():
            raise MarketDataAcquisitionCancelled(
                f"Historical market-data acquisition cancelled for '{symbol}' "
                f"after {index - 1}/{total} requests"
            )

        logger.info(
            "Historical request | symbol=%s chunk=%d/%d window=%s→%s",
            symbol,
            index,
            total,
            chunk_start.isoformat(),
            chunk_end.isoformat(),
        )
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
    logger.info(
        "Historical source complete | symbol=%s requests=%d bars=%d",
        symbol,
        total,
        len(records),
    )
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
    *,
    max_span: timedelta,
    session: tuple[time, time, ZoneInfo] | None = None,
) -> Iterable[tuple[datetime, datetime]]:
    """Yield broker request windows respecting IB duration limits.

    When a trading-session window is supplied, the full range is first split
    into one window per weekday session (mirroring the legacy day-by-day RTH
    collection). Every window is then sub-divided so that no single request
    exceeds ``max_span``, keeping each ``durationStr`` valid (in particular
    never exceeding 86400 seconds when expressed in the ``S`` unit).
    """

    start_utc = _ensure_utc(start)
    end_utc = _ensure_utc(end)

    if session is not None:
        outer_ranges: Iterable[tuple[datetime, datetime]] = _session_day_ranges(
            start_utc, end_utc, session
        )
    else:
        outer_ranges = ((start_utc, end_utc),)

    for outer_start, outer_end in outer_ranges:
        yield from _split_span(outer_start, outer_end, max_span)


def _split_span(
    start: datetime,
    end: datetime,
    max_span: timedelta,
) -> Iterable[tuple[datetime, datetime]]:
    if start >= end:
        if start == end:
            yield start, end
        return
    current = start
    while current < end:
        chunk_end = min(end, current + max_span)
        yield current, chunk_end
        current = chunk_end


def _session_day_ranges(
    start: datetime,
    end: datetime,
    session: tuple[time, time, ZoneInfo],
) -> Iterable[tuple[datetime, datetime]]:
    session_start, session_end, tz = session
    day = start.astimezone(tz).date()
    last_day = end.astimezone(tz).date()
    while day <= last_day:
        # Skip weekends; IB returns no RTH bars and they only waste pacing.
        if day.weekday() < 5:
            window_start = datetime.combine(
                day, session_start, tzinfo=tz
            ).astimezone(UTC)
            window_end = datetime.combine(day, session_end, tzinfo=tz).astimezone(UTC)
            clamped_start = max(window_start, start)
            clamped_end = min(window_end, end)
            if clamped_start < clamped_end:
                yield clamped_start, clamped_end
        day += timedelta(days=1)


def _is_intraday_frequency(frequency: DataFrequency) -> bool:
    """Return whether a frequency produces intraday bars (sub-daily)."""

    return frequency not in {DataFrequency.DAY_1, DataFrequency.IRREGULAR}


def _max_request_span(frequency: DataFrequency) -> timedelta:
    """Return the maximum duration permitted per request for a bar size.

    Values follow the Interactive Brokers historical-data step-size table so
    each request's duration string is independently valid.
    """

    if frequency in {DataFrequency.TICK, DataFrequency.SECOND_1}:
        return timedelta(seconds=1800)
    if frequency == DataFrequency.SECOND_5:
        return timedelta(seconds=3600)
    if frequency == DataFrequency.SECOND_10:
        return timedelta(seconds=14400)
    if frequency == DataFrequency.SECOND_30:
        return timedelta(seconds=28800)
    if frequency == DataFrequency.MINUTE_1:
        return timedelta(days=1)
    if frequency in {DataFrequency.MINUTE_5, DataFrequency.MINUTE_15}:
        return timedelta(days=7)
    if frequency == DataFrequency.HOUR_1:
        return timedelta(days=30)
    if frequency == DataFrequency.DAY_1:
        return timedelta(days=365)
    return timedelta(days=1)


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
    "MarketDataAcquisitionCancelled",
    "MarketDataAcquisitionError",
    "SymbolAcquisitionReport",
]
