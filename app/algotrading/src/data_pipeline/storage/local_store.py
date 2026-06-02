"""Canonical file-backed store for sourced market and news data."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import pandas as pd
from algotrading.src.data_pipeline import (
    DataBatch,
    DataFrequency,
    DataRecord,
    DataType,
    NewsRecord,
)
from algotrading.src.data_pipeline.sources.exceptions import (
    DataNotFoundError,
    DataSourceError,
    DataValidationError,
)

MARKET_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "trade_count",
    "source_id",
    "frequency",
)


class _ProviderConfigLike(Protocol):
    """Provider configuration fields required for local cache lookup."""

    provider: object


class _TrainingDataRequestLike(Protocol):
    """Training data request fields required for local cache lookup."""

    symbols: Sequence[str]
    start_time: datetime
    end_time: datetime
    frequency: DataFrequency
    market_source: _ProviderConfigLike
    news_source: _ProviderConfigLike
    cache_policy: object


class LocalDataStoreError(DataSourceError):
    """Base error for local sourced-data storage operations."""


class LocalDataValidationError(LocalDataStoreError, DataValidationError):
    """Raised when local sourced data is malformed or incomplete."""


class LocalDataCacheMissError(LocalDataStoreError, DataNotFoundError):
    """Raised when requested local data does not exist or lacks coverage."""


@dataclass(frozen=True, slots=True)
class DataManifest:
    """Manifest metadata for a locally persisted sourced-data artifact."""

    data_type: DataType
    provider: str
    symbols: tuple[str, ...]
    requested_start: datetime
    requested_end: datetime
    actual_start: datetime | None
    actual_end: datetime | None
    frequency: DataFrequency | None
    record_count: int
    cache_policy: str
    source_file_paths: tuple[str, ...]
    data_file_paths: tuple[str, ...]
    created_at: datetime
    manifest_path: Path | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize manifest metadata to a JSON-compatible dictionary."""

        return {
            "data_type": self.data_type.value,
            "provider": self.provider,
            "symbols": list(self.symbols),
            "requested_range": {
                "start": _format_datetime(self.requested_start),
                "end": _format_datetime(self.requested_end),
            },
            "actual_range": {
                "start": _format_datetime(self.actual_start),
                "end": _format_datetime(self.actual_end),
            },
            "frequency": self.frequency.value if self.frequency else None,
            "record_count": self.record_count,
            "cache_policy": self.cache_policy,
            "source_file_paths": list(self.source_file_paths),
            "data_file_paths": list(self.data_file_paths),
            "created_at": _format_datetime(self.created_at),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        manifest_path: str | Path | None = None,
    ) -> DataManifest:
        """Deserialize manifest metadata from a JSON payload."""

        requested_range = _mapping_value(payload.get("requested_range"))
        actual_range = _mapping_value(payload.get("actual_range"))
        frequency_value = payload.get("frequency")

        return cls(
            data_type=_parse_data_type(payload.get("data_type")),
            provider=_slug(payload.get("provider")),
            symbols=tuple(_normalize_symbols(payload.get("symbols"))),
            requested_start=_parse_utc_datetime(requested_range.get("start")),
            requested_end=_parse_utc_datetime(requested_range.get("end")),
            actual_start=_parse_optional_utc_datetime(actual_range.get("start")),
            actual_end=_parse_optional_utc_datetime(actual_range.get("end")),
            frequency=(
                _parse_frequency(frequency_value)
                if frequency_value is not None
                else None
            ),
            record_count=int(payload.get("record_count") or 0),
            cache_policy=_policy_value(payload.get("cache_policy")),
            source_file_paths=tuple(_string_sequence(payload.get("source_file_paths"))),
            data_file_paths=tuple(_string_sequence(payload.get("data_file_paths"))),
            created_at=_parse_utc_datetime(payload.get("created_at")),
            manifest_path=Path(manifest_path) if manifest_path is not None else None,
        )


@dataclass(frozen=True, slots=True)
class StoredMarketData:
    """Loaded market data plus its local storage metadata."""

    symbol: str
    batch: DataBatch
    manifest: DataManifest
    data_path: Path
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class StoredNewsData:
    """Loaded news records plus their local storage metadata."""

    symbol: str
    records: list[NewsRecord]
    manifest: DataManifest
    data_path: Path
    manifest_path: Path


class LocalDataStore:
    """Read and write canonical sourced market bars, news records, and manifests."""

    MARKET_FILENAME = "market.csv"
    NEWS_FILENAME = "news.jsonl"
    MANIFEST_FILENAME = "manifest.json"

    def __init__(self, root: str | Path | None = None) -> None:
        """Initialize the store with a sourced-data root directory."""

        if root is None:
            repo_root = Path(__file__).resolve().parents[5]
            root = repo_root / "data" / "sourced"
        self.root = Path(root)

    def write_market_batch(
        self,
        batch: DataBatch,
        *,
        provider: object,
        frequency: object | None = None,
        requested_start: datetime | None = None,
        requested_end: datetime | None = None,
        cache_policy: object = "prefer_cache",
        source_file_paths: Sequence[str | Path] | None = None,
    ) -> DataManifest:
        """Persist one market batch as canonical CSV plus manifest."""

        if batch.data_type != DataType.MARKET_BAR:
            raise LocalDataValidationError(
                "market batches must use DataType.MARKET_BAR"
            )
        if not batch.records:
            raise LocalDataValidationError(
                f"Cannot write empty market batch for symbol '{batch.symbol}'"
            )

        symbol = _normalize_symbol(batch.symbol)
        resolved_frequency = _resolve_market_frequency(batch, frequency)
        directory = self._market_directory(
            provider=provider,
            symbol=symbol,
            frequency=resolved_frequency,
        )
        data_path = directory / self.MARKET_FILENAME
        manifest_path = directory / self.MANIFEST_FILENAME

        rows = _market_rows(batch.records, symbol=symbol, frequency=resolved_frequency)
        frame = pd.DataFrame(rows, columns=MARKET_COLUMNS)
        frame = frame.sort_values("timestamp").drop_duplicates(
            subset=["symbol", "timestamp"], keep="last"
        )
        actual_start = _parse_utc_datetime(frame.iloc[0]["timestamp"])
        actual_end = _parse_utc_datetime(frame.iloc[-1]["timestamp"])

        directory.mkdir(parents=True, exist_ok=True)
        self._atomic_write_dataframe(frame, data_path)

        manifest = DataManifest(
            data_type=DataType.MARKET_BAR,
            provider=_slug(provider),
            symbols=(symbol,),
            requested_start=_ensure_utc(requested_start or batch.start_time),
            requested_end=_ensure_utc(requested_end or batch.end_time),
            actual_start=actual_start,
            actual_end=actual_end,
            frequency=resolved_frequency,
            record_count=len(frame),
            cache_policy=_policy_value(cache_policy),
            source_file_paths=tuple(str(path) for path in source_file_paths or ()),
            data_file_paths=(str(data_path),),
            created_at=datetime.now(tz=UTC),
            manifest_path=manifest_path,
        )
        return self.write_manifest(manifest, manifest_path)

    def find_market_data(
        self,
        request: _TrainingDataRequestLike,
        symbols: Sequence[str] | None = None,
    ) -> dict[str, StoredMarketData]:
        """Load cached market data covering the requested symbols and range."""

        requested_symbols = _normalize_symbols(symbols or request.symbols)
        if not requested_symbols:
            raise LocalDataValidationError("market cache lookup requires symbols")

        provider = request.market_source.provider
        frequency = _parse_frequency(request.frequency)
        results: dict[str, StoredMarketData] = {}

        for symbol in requested_symbols:
            directory = self._market_directory(
                provider=provider,
                symbol=symbol,
                frequency=frequency,
            )
            data_path = directory / self.MARKET_FILENAME
            manifest_path = directory / self.MANIFEST_FILENAME
            manifest = self.read_manifest(manifest_path)
            self._require_manifest_coverage(
                manifest=manifest,
                request=request,
                symbol=symbol,
                provider=provider,
                frequency=frequency,
            )
            batch = self._read_market_batch(
                data_path=data_path,
                symbol=symbol,
                start=request.start_time,
                end=request.end_time,
                frequency=frequency,
            )
            results[symbol] = StoredMarketData(
                symbol=symbol,
                batch=batch,
                manifest=manifest,
                data_path=data_path,
                manifest_path=manifest_path,
            )
        return results

    def write_news_records(
        self,
        records: Sequence[NewsRecord],
        *,
        provider: object,
        symbol: str,
        requested_start: datetime,
        requested_end: datetime,
        cache_policy: object = "prefer_cache",
        source_file_paths: Sequence[str | Path] | None = None,
    ) -> DataManifest:
        """Persist news records for one symbol as JSONL plus manifest."""

        normalized_symbol = _normalize_symbol(symbol)
        relevant_records = _dedupe_news_records(
            record
            for record in records
            if _news_matches_symbol(record, normalized_symbol)
        )
        if not relevant_records:
            raise LocalDataValidationError(
                f"Cannot write empty news record set for symbol '{normalized_symbol}'"
            )

        directory = self._news_directory(provider=provider, symbol=normalized_symbol)
        data_path = directory / self.NEWS_FILENAME
        manifest_path = directory / self.MANIFEST_FILENAME
        directory.mkdir(parents=True, exist_ok=True)

        relevant_records.sort(
            key=lambda item: (_ensure_utc(item.timestamp), item.news_id)
        )
        self._atomic_write_jsonl(
            (_news_record_to_dict(record) for record in relevant_records),
            data_path,
        )

        actual_start = _ensure_utc(relevant_records[0].timestamp)
        actual_end = _ensure_utc(relevant_records[-1].timestamp)
        manifest = DataManifest(
            data_type=DataType.NEWS_TEXT,
            provider=_slug(provider),
            symbols=(normalized_symbol,),
            requested_start=_ensure_utc(requested_start),
            requested_end=_ensure_utc(requested_end),
            actual_start=actual_start,
            actual_end=actual_end,
            frequency=DataFrequency.IRREGULAR,
            record_count=len(relevant_records),
            cache_policy=_policy_value(cache_policy),
            source_file_paths=tuple(str(path) for path in source_file_paths or ()),
            data_file_paths=(str(data_path),),
            created_at=datetime.now(tz=UTC),
            manifest_path=manifest_path,
        )
        return self.write_manifest(manifest, manifest_path)

    def find_news_data(
        self,
        request: _TrainingDataRequestLike,
        symbols: Sequence[str] | None = None,
    ) -> dict[str, StoredNewsData]:
        """Load cached news records covering the requested symbols and range."""

        requested_symbols = _normalize_symbols(symbols or request.symbols)
        if not requested_symbols:
            raise LocalDataValidationError("news cache lookup requires symbols")

        provider = request.news_source.provider
        if _slug(provider) == "none":
            return {}

        results: dict[str, StoredNewsData] = {}
        for symbol in requested_symbols:
            directory = self._news_directory(provider=provider, symbol=symbol)
            data_path = directory / self.NEWS_FILENAME
            manifest_path = directory / self.MANIFEST_FILENAME
            manifest = self.read_manifest(manifest_path)
            self._require_manifest_coverage(
                manifest=manifest,
                request=request,
                symbol=symbol,
                provider=provider,
                frequency=DataFrequency.IRREGULAR,
            )
            records = self._read_news_records(
                data_path=data_path,
                symbol=symbol,
                start=request.start_time,
                end=request.end_time,
            )
            results[symbol] = StoredNewsData(
                symbol=symbol,
                records=records,
                manifest=manifest,
                data_path=data_path,
                manifest_path=manifest_path,
            )
        return results

    def write_manifest(
        self,
        manifest: DataManifest,
        manifest_path: str | Path,
    ) -> DataManifest:
        """Atomically write one manifest JSON file and return its persisted view."""

        path = Path(manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write_json(manifest.to_dict(), path)
        return replace(manifest, manifest_path=path)

    def read_manifest(self, manifest_path: str | Path) -> DataManifest:
        """Read one manifest JSON file."""

        path = Path(manifest_path)
        if not path.exists():
            raise LocalDataCacheMissError(f"Local data manifest not found: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise LocalDataValidationError(
                f"Manifest is not valid JSON: {path}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise LocalDataValidationError(f"Manifest root must be an object: {path}")
        return DataManifest.from_dict(payload, manifest_path=path)

    def _market_directory(
        self,
        *,
        provider: object,
        symbol: str,
        frequency: DataFrequency,
    ) -> Path:
        return self.root / "market" / _slug(provider) / symbol / frequency.value

    def _news_directory(self, *, provider: object, symbol: str) -> Path:
        return self.root / "news" / _slug(provider) / symbol

    def _read_market_batch(
        self,
        *,
        data_path: Path,
        symbol: str,
        start: datetime,
        end: datetime,
        frequency: DataFrequency,
    ) -> DataBatch:
        if not data_path.exists():
            raise LocalDataCacheMissError(
                f"Local market data file not found: {data_path}"
            )
        try:
            frame = pd.read_csv(data_path)
        except Exception as exc:
            raise LocalDataValidationError(
                f"Failed to read local market CSV {data_path}: {exc}"
            ) from exc

        missing_columns = [column for column in MARKET_COLUMNS if column not in frame]
        if missing_columns:
            raise LocalDataValidationError(
                f"Local market CSV {data_path} is missing columns: "
                f"{', '.join(missing_columns)}"
            )

        timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.assign(timestamp=timestamps)
        frame = frame.dropna(subset=["timestamp"])
        frame = frame[frame["symbol"].astype(str).str.upper() == symbol]
        start_utc = _ensure_utc(start)
        end_utc = _ensure_utc(end)
        frame = frame[
            (frame["timestamp"] >= pd.Timestamp(start_utc))
            & (frame["timestamp"] <= pd.Timestamp(end_utc))
        ]
        frame = frame.sort_values("timestamp")
        if frame.empty:
            raise LocalDataCacheMissError(
                f"Local market cache has no rows for {symbol} between "
                f"{start_utc.isoformat()} and {end_utc.isoformat()} at {data_path}"
            )

        records = [
            _record_from_market_row(row, symbol, frequency)
            for row in frame.itertuples(index=False)
        ]
        return DataBatch(
            records=records,
            start_time=records[0].timestamp,
            end_time=records[-1].timestamp,
            data_type=DataType.MARKET_BAR,
            symbol=symbol,
        )

    def _read_news_records(
        self,
        *,
        data_path: Path,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[NewsRecord]:
        if not data_path.exists():
            raise LocalDataCacheMissError(
                f"Local news data file not found: {data_path}"
            )

        records: list[NewsRecord] = []
        start_utc = _ensure_utc(start)
        end_utc = _ensure_utc(end)
        try:
            with data_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    payload = json.loads(stripped)
                    if not isinstance(payload, Mapping):
                        raise LocalDataValidationError(
                            f"News JSONL line {line_number} must be an object: "
                            f"{data_path}"
                        )
                    record = _news_record_from_dict(payload)
                    record_time = _ensure_utc(record.timestamp)
                    if (
                        _news_matches_symbol(record, symbol)
                        and start_utc <= record_time <= end_utc
                    ):
                        records.append(record)
        except json.JSONDecodeError as exc:
            raise LocalDataValidationError(
                f"News JSONL contains invalid JSON at {data_path}: {exc}"
            ) from exc

        if not records:
            raise LocalDataCacheMissError(
                f"Local news cache has no records for {symbol} between "
                f"{start_utc.isoformat()} and {end_utc.isoformat()} at {data_path}"
            )
        records.sort(key=lambda item: (_ensure_utc(item.timestamp), item.news_id))
        return records

    def _require_manifest_coverage(
        self,
        *,
        manifest: DataManifest,
        request: _TrainingDataRequestLike,
        symbol: str,
        provider: object,
        frequency: DataFrequency,
    ) -> None:
        requested_start = _ensure_utc(request.start_time)
        requested_end = _ensure_utc(request.end_time)
        if symbol not in manifest.symbols:
            raise LocalDataCacheMissError(
                f"Local manifest for provider '{_slug(provider)}' does not include "
                f"symbol '{symbol}'"
            )
        if manifest.actual_start is None or manifest.actual_end is None:
            raise LocalDataCacheMissError(
                f"Local cache for {symbol} has no actual coverage range in manifest"
            )
        coverage_start = manifest.actual_start
        coverage_end = manifest.actual_end
        if manifest.data_type == DataType.NEWS_TEXT:
            coverage_start = manifest.requested_start
            coverage_end = manifest.requested_end
        if coverage_start > requested_start or coverage_end < requested_end:
            raise LocalDataCacheMissError(
                f"Local cache for {symbol} from provider '{_slug(provider)}' does "
                f"not cover {requested_start.isoformat()} to "
                f"{requested_end.isoformat()}; actual coverage is "
                f"{coverage_start.isoformat()} to "
                f"{coverage_end.isoformat()}"
            )
        if (
            manifest.data_type == DataType.MARKET_BAR
            and manifest.frequency != frequency
        ):
            actual_frequency = (
                manifest.frequency.value if manifest.frequency else "unknown"
            )
            raise LocalDataCacheMissError(
                f"Local market cache for {symbol} has frequency {actual_frequency}, "
                f"requested {frequency.value}"
            )

    def _atomic_write_dataframe(self, frame: pd.DataFrame, path: Path) -> None:
        temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            frame.to_csv(temporary_path, index=False)
            temporary_path.replace(path)
        finally:
            _unlink_if_exists(temporary_path)

    def _atomic_write_jsonl(
        self,
        rows: Iterable[Mapping[str, object]],
        path: Path,
    ) -> None:
        temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(dict(row), sort_keys=True))
                    handle.write("\n")
            temporary_path.replace(path)
        finally:
            _unlink_if_exists(temporary_path)

    def _atomic_write_json(self, payload: Mapping[str, object], path: Path) -> None:
        temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary_path.write_text(
                json.dumps(dict(payload), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            temporary_path.replace(path)
        finally:
            _unlink_if_exists(temporary_path)


def _market_rows(
    records: Sequence[DataRecord],
    *,
    symbol: str,
    frequency: DataFrequency,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        if record.data_type != DataType.MARKET_BAR:
            raise LocalDataValidationError(
                "market CSV can only contain MARKET_BAR records"
            )
        record_symbol = _normalize_symbol(record.symbol)
        if record_symbol != symbol:
            raise LocalDataValidationError(
                f"market batch symbol '{symbol}' contains record for '{record_symbol}'"
            )
        payload = record.payload
        rows.append(
            {
                "timestamp": _format_datetime(record.timestamp),
                "symbol": symbol,
                "open": _required_float(payload, "open"),
                "high": _required_float(payload, "high"),
                "low": _required_float(payload, "low"),
                "close": _required_float(payload, "close"),
                "volume": _required_int(payload, "volume"),
                "vwap": _optional_float(_first_payload_value(payload, "vwap", "wap")),
                "trade_count": _optional_int(
                    _first_payload_value(payload, "trade_count", "count")
                ),
                "source_id": record.source_id or "",
                "frequency": frequency.value,
            }
        )
    return rows


def _record_from_market_row(
    row: object,
    symbol: str,
    frequency: DataFrequency,
) -> DataRecord:
    payload = {
        "open": float(getattr(row, "open")),
        "high": float(getattr(row, "high")),
        "low": float(getattr(row, "low")),
        "close": float(getattr(row, "close")),
        "volume": int(getattr(row, "volume")),
        "vwap": _optional_float(getattr(row, "vwap")),
        "trade_count": _optional_int(getattr(row, "trade_count")),
    }
    return DataRecord(
        timestamp=_parse_utc_datetime(getattr(row, "timestamp")),
        data_type=DataType.MARKET_BAR,
        symbol=symbol,
        payload=payload,
        source_id=_optional_text(getattr(row, "source_id")),
        frequency=frequency,
    )


def _news_record_to_dict(record: NewsRecord) -> dict[str, object]:
    return {
        "timestamp": _format_datetime(record.timestamp),
        "headline": record.headline,
        "body": record.body,
        "source": record.source,
        "symbols": [_normalize_symbol(symbol) for symbol in record.symbols],
        "categories": list(record.categories),
        "sentiment_score": record.sentiment_score,
        "url": record.url,
        "news_id": record.news_id,
    }


def _news_record_from_dict(payload: Mapping[str, object]) -> NewsRecord:
    return NewsRecord(
        timestamp=_parse_utc_datetime(payload.get("timestamp")),
        headline=str(payload.get("headline") or ""),
        body=str(payload["body"]) if payload.get("body") is not None else None,
        source=str(payload.get("source") or ""),
        symbols=_normalize_symbols(payload.get("symbols")),
        categories=_string_sequence(payload.get("categories")),
        sentiment_score=_optional_float(payload.get("sentiment_score")),
        url=str(payload["url"]) if payload.get("url") is not None else None,
        news_id=str(payload.get("news_id") or ""),
    )


def _dedupe_news_records(records: Iterable[NewsRecord]) -> list[NewsRecord]:
    by_id: dict[str, NewsRecord] = {}
    for record in records:
        by_id[record.news_id] = record
    return list(by_id.values())


def _news_matches_symbol(record: NewsRecord, symbol: str) -> bool:
    return symbol in {_normalize_symbol(value) for value in record.symbols}


def _resolve_market_frequency(
    batch: DataBatch,
    frequency: object | None,
) -> DataFrequency:
    if frequency is not None:
        return _parse_frequency(frequency)
    for record in batch.records:
        if record.frequency is not None:
            return _parse_frequency(record.frequency)
    raise LocalDataValidationError(
        "market batch records must include frequency or frequency must be supplied"
    )


def _required_float(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if value is None:
        raise LocalDataValidationError(f"market payload missing required '{key}'")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise LocalDataValidationError(
            f"market payload '{key}' must be numeric"
        ) from exc


def _required_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if value is None:
        raise LocalDataValidationError(f"market payload missing required '{key}'")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise LocalDataValidationError(
            f"market payload '{key}' must be an integer"
        ) from exc


def _optional_float(value: object) -> float | None:
    if _is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise LocalDataValidationError(
            "optional numeric value must be numeric"
        ) from exc


def _optional_int(value: object) -> int | None:
    if _is_missing(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise LocalDataValidationError(
            "optional integer value must be an integer"
        ) from exc


def _optional_text(value: object) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text or None


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _first_payload_value(payload: Mapping[str, object], *keys: str) -> object | None:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def _mapping_value(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return value


def _string_sequence(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item) for item in value if str(item)]
    return [str(value)] if str(value) else []


def _normalize_symbols(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        candidates = value.replace(";", ",").split(",")
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        candidates = [str(item) for item in value]
    else:
        candidates = [str(value)]
    normalized = [
        _normalize_symbol(candidate)
        for candidate in candidates
        if str(candidate).strip()
    ]
    return list(dict.fromkeys(normalized))


def _normalize_symbol(value: object) -> str:
    symbol = str(value).strip().upper()
    if not symbol:
        raise LocalDataValidationError("symbol must be non-empty")
    return symbol


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _parse_utc_datetime(value: object) -> datetime:
    parsed = _parse_optional_utc_datetime(value)
    if parsed is None:
        raise LocalDataValidationError("datetime value is required")
    return parsed


def _parse_optional_utc_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return _ensure_utc(value.to_pydatetime())
    if isinstance(value, datetime):
        return _ensure_utc(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LocalDataValidationError(f"Invalid datetime value '{value}'") from exc
    return _ensure_utc(parsed)


def _format_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return _ensure_utc(value).isoformat()


def _parse_data_type(value: object) -> DataType:
    if isinstance(value, DataType):
        return value
    try:
        return DataType(str(value))
    except ValueError:
        member_name = str(value).upper()
        if member_name in DataType.__members__:
            return DataType[member_name]
    raise LocalDataValidationError(f"Unsupported data_type '{value}'")


def _parse_frequency(value: object) -> DataFrequency:
    if isinstance(value, DataFrequency):
        return value
    try:
        return DataFrequency(str(value))
    except ValueError:
        member_name = str(value).upper()
        if member_name in DataFrequency.__members__:
            return DataFrequency[member_name]
    raise LocalDataValidationError(f"Unsupported data frequency '{value}'")


def _slug(value: object) -> str:
    if hasattr(value, "value"):
        value = getattr(value, "value")
    slug = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    if slug == "interactive_brokers":
        return "ib"
    if not slug:
        raise LocalDataValidationError("provider must be non-empty")
    return slug


def _policy_value(value: object) -> str:
    if hasattr(value, "value"):
        value = getattr(value, "value")
    policy = str(value or "prefer_cache").strip().lower().replace("-", "_")
    if policy not in {"prefer_cache", "refresh", "require_cache"}:
        raise LocalDataValidationError(
            "cache_policy must be one of prefer_cache, refresh, or require_cache"
        )
    return policy


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return


__all__ = [
    "DataManifest",
    "LocalDataCacheMissError",
    "LocalDataStore",
    "LocalDataStoreError",
    "LocalDataValidationError",
    "MARKET_COLUMNS",
    "StoredMarketData",
    "StoredNewsData",
]
