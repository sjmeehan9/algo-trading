"""File-backed data source implementation for CSV market data."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterator
from zoneinfo import ZoneInfo

import pandas as pd
from algotrading.src.data_pipeline.sources.base import DataSource
from algotrading.src.data_pipeline.sources.exceptions import (
    DataNotFoundError,
    DataSourceConnectionError,
    DataValidationError,
)
from algotrading.src.data_pipeline.types import (
    DataBatch,
    DataFrequency,
    DataRecord,
    DataType,
    SourceMetadata,
)


@dataclass(frozen=True, slots=True)
class _IndexedFile:
    """Indexed metadata for one source file."""

    symbol: str
    exchange: str
    file_date: date
    path: Path


class FileSource(DataSource):
    """CSV-backed data source supporting batch and stream loading.

    Args:
        base_path: Directory containing source files.
        file_pattern: Naming convention for source files.
        date_format: Date token format used in filenames.
        columns_config: Optional column validation rules.
        trim_percentage: Proportion of rows removed from each file's edges.
        timezone: Timezone for timestamp localization.
        stream_delay_seconds: Delay between streamed records.
    """

    DEFAULT_COLUMNS_CONFIG: dict[str, dict[str, object]] = {
        "date": {"required": True, "numeric": False},
        "open": {"required": True, "numeric": True},
        "high": {"required": True, "numeric": True},
        "low": {"required": True, "numeric": True},
        "close": {"required": True, "numeric": True},
        "volume": {"required": True, "numeric": True},
        "wap": {"required": True, "numeric": True},
        "count": {"required": True, "numeric": True},
    }

    def __init__(
        self,
        base_path: str,
        file_pattern: str = "{symbol}_{exchange}_{date}.csv",
        date_format: str = "%Y%m%d",
        columns_config: dict[str, dict[str, object]] | None = None,
        trim_percentage: float = 0.038,
        timezone: str = "US/Eastern",
        stream_delay_seconds: float = 0.0,
    ) -> None:
        self._base_path = Path(base_path)
        self._file_pattern = file_pattern
        self._date_format = date_format
        self._columns_config = columns_config or self.DEFAULT_COLUMNS_CONFIG.copy()
        self._trim_percentage = trim_percentage
        self._timezone = timezone
        self._stream_delay_seconds = stream_delay_seconds

        self._zoneinfo = ZoneInfo(self._timezone)
        self._filename_regex = self._build_filename_regex(self._file_pattern)
        self._connected = False

        self._files: list[_IndexedFile] = []
        self._files_by_symbol: dict[str, list[_IndexedFile]] = {}

    @property
    def source_id(self) -> str:
        """Return stable identifier for this source instance."""

        return f"file_source:{self._base_path.resolve()}"

    @property
    def metadata(self) -> SourceMetadata:
        """Return source capabilities and runtime configuration."""

        return SourceMetadata(
            source_id=self.source_id,
            source_type="file",
            supported_types=[DataType.MARKET_BAR],
            supported_frequencies=[
                DataFrequency.TICK,
                DataFrequency.SECOND_1,
                DataFrequency.SECOND_5,
                DataFrequency.MINUTE_1,
                DataFrequency.MINUTE_5,
                DataFrequency.DAY_1,
            ],
            config={
                "base_path": str(self._base_path),
                "file_pattern": self._file_pattern,
                "date_format": self._date_format,
                "trim_percentage": self._trim_percentage,
                "timezone": self._timezone,
            },
        )

    @property
    def is_connected(self) -> bool:
        """Return whether the source has been connected and indexed."""

        return self._connected

    def connect(self) -> None:
        """Index files in ``base_path`` and initialize source state."""

        if not self._base_path.exists() or not self._base_path.is_dir():
            raise DataSourceConnectionError(
                f"File source path does not exist or is not a directory: {self._base_path}"
            )

        indexed_files: list[_IndexedFile] = []
        for path in sorted(self._base_path.glob("*.csv")):
            symbol, exchange, parsed_date = self._parse_filename(path.name)
            if symbol is None or exchange is None or parsed_date is None:
                continue
            indexed_files.append(
                _IndexedFile(
                    symbol=symbol,
                    exchange=exchange,
                    file_date=parsed_date,
                    path=path,
                )
            )

        self._files = indexed_files
        self._files_by_symbol = {}
        for file_info in self._files:
            self._files_by_symbol.setdefault(file_info.symbol, []).append(file_info)

        for symbol in self._files_by_symbol:
            self._files_by_symbol[symbol].sort(
                key=lambda item: (item.file_date, item.path)
            )

        self._connected = True

    def disconnect(self) -> None:
        """Reset in-memory index and connection state."""

        self._connected = False
        self._files = []
        self._files_by_symbol = {}

    def fetch_batch(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        data_type: DataType = DataType.MARKET_BAR,
    ) -> DataBatch:
        """Load all records for ``symbol`` between ``start`` and ``end``.

        Raises:
            DataNotFoundError: If no files/records match the request.
            DataValidationError: If CSV content is invalid.
        """

        self._ensure_connected()
        if data_type != DataType.MARKET_BAR:
            raise DataValidationError(
                "FileSource currently supports DataType.MARKET_BAR only"
            )
        if start > end:
            raise DataValidationError("start must be less than or equal to end")

        if symbol not in self._files_by_symbol:
            raise DataNotFoundError(f"No files indexed for symbol: {symbol}")

        start_date = start.date()
        end_date = end.date()
        matching_files = [
            item
            for item in self._files_by_symbol[symbol]
            if start_date <= item.file_date <= end_date
        ]
        if not matching_files:
            raise DataNotFoundError(
                f"No files found for symbol {symbol} between {start_date} and {end_date}"
            )

        records: list[DataRecord] = []
        for file_info in matching_files:
            loaded = self._load_csv(file_info.path)
            filtered = loaded[(loaded["date"] >= start) & (loaded["date"] <= end)]
            if filtered.empty:
                continue
            records.extend(self._dataframe_to_records(filtered, symbol))

        if not records:
            raise DataNotFoundError(
                f"No records found for symbol {symbol} between {start.isoformat()} and {end.isoformat()}"
            )

        records.sort(key=lambda item: item.timestamp)
        return DataBatch(
            records=records,
            start_time=records[0].timestamp,
            end_time=records[-1].timestamp,
            data_type=DataType.MARKET_BAR,
            symbol=symbol,
        )

    def fetch_stream(
        self,
        symbol: str,
        data_type: DataType = DataType.MARKET_BAR,
    ) -> Iterator[DataRecord]:
        """Yield records for ``symbol`` sequentially from indexed files."""

        self._ensure_connected()

        if symbol not in self._files_by_symbol:
            raise DataNotFoundError(f"No files indexed for symbol: {symbol}")

        available_dates = self.get_available_dates(symbol)
        if not available_dates:
            raise DataNotFoundError(f"No dates available for symbol: {symbol}")

        start = datetime.combine(
            min(available_dates), datetime.min.time(), tzinfo=self._zoneinfo
        )
        end = datetime.combine(
            max(available_dates), datetime.max.time(), tzinfo=self._zoneinfo
        )
        batch = self.fetch_batch(
            symbol=symbol, start=start, end=end, data_type=data_type
        )

        def _record_iterator() -> Iterator[DataRecord]:
            for record in batch:
                yield record
                if self._stream_delay_seconds > 0:
                    time.sleep(self._stream_delay_seconds)

        return _record_iterator()

    def get_available_symbols(self) -> list[str]:
        """Return all symbols indexed from file names."""

        self._ensure_connected()
        return sorted(self._files_by_symbol.keys())

    def get_available_dates(self, symbol: str) -> list[date]:
        """Return sorted dates with available data for ``symbol``."""

        self._ensure_connected()
        return sorted(
            {item.file_date for item in self._files_by_symbol.get(symbol, [])}
        )

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise DataSourceConnectionError(
                "FileSource is not connected. Call connect() before fetching data."
            )

    def _build_filename_regex(self, pattern: str) -> re.Pattern[str]:
        escaped = re.escape(pattern)
        escaped = escaped.replace(r"\{symbol\}", r"(?P<symbol>[^_]+)")
        escaped = escaped.replace(r"\{exchange\}", r"(?P<exchange>[^_]+)")
        escaped = escaped.replace(r"\{date\}", r"(?P<date>\d+)")
        return re.compile(f"^{escaped}$")

    def _parse_filename(
        self, filename: str
    ) -> tuple[str | None, str | None, date | None]:
        """Extract ``(symbol, exchange, date)`` from a CSV filename."""

        match = self._filename_regex.match(filename)
        if match is None:
            return None, None, None

        symbol = match.group("symbol")
        exchange = match.group("exchange")
        try:
            parsed_date = datetime.strptime(
                match.group("date"), self._date_format
            ).date()
        except ValueError:
            return None, None, None

        return symbol, exchange, parsed_date

    def _load_csv(self, filepath: Path) -> pd.DataFrame:
        """Load and validate one CSV file, including trim behavior."""

        try:
            data = pd.read_csv(filepath)
        except Exception as exc:
            raise DataValidationError(
                f"Failed to read CSV file {filepath}: {exc}"
            ) from exc

        self._validate_csv_columns(data)

        if self._trim_percentage > 0 and len(data) > 2:
            rows_to_drop = int(len(data) * self._trim_percentage / 2)
            if rows_to_drop > 0 and len(data) > (rows_to_drop * 2):
                data = data.iloc[rows_to_drop:-rows_to_drop].copy()

        parsed_dates: list[datetime] = []
        for raw_value in data["date"].astype(str).tolist():
            parsed_dates.append(self._parse_timestamp(raw_value))

        data = data.copy()
        data["date"] = parsed_dates
        return data

    def _parse_timestamp(self, raw_value: str) -> datetime:
        value = raw_value.strip()
        legacy_suffix = f" {self._timezone}"
        if value.endswith(legacy_suffix):
            naive_part = value[: -len(legacy_suffix)]
            parsed = datetime.strptime(naive_part, "%Y%m%d %H:%M:%S")
            return parsed.replace(tzinfo=self._zoneinfo)

        parsed_ts = pd.to_datetime(value, errors="raise")
        dt_value = parsed_ts.to_pydatetime()
        if dt_value.tzinfo is None:
            return dt_value.replace(tzinfo=self._zoneinfo)
        return dt_value

    def _validate_csv_columns(self, dataframe: pd.DataFrame) -> None:
        """Validate presence and basic types of required CSV columns."""

        required_columns = [
            column
            for column, rules in self._columns_config.items()
            if bool(rules.get("required", False))
        ]
        missing_columns = [
            column for column in required_columns if column not in dataframe.columns
        ]
        if missing_columns:
            raise DataValidationError(
                f"Missing required columns: {', '.join(sorted(missing_columns))}"
            )

        numeric_columns = [
            column
            for column, rules in self._columns_config.items()
            if bool(rules.get("numeric", False)) and column in dataframe.columns
        ]
        for column in numeric_columns:
            coerced = pd.to_numeric(dataframe[column], errors="coerce")
            if coerced.isna().any():
                raise DataValidationError(
                    f"Column '{column}' contains non-numeric values"
                )

    def _dataframe_to_records(
        self, dataframe: pd.DataFrame, symbol: str
    ) -> list[DataRecord]:
        """Convert validated dataframe rows into pipeline records."""

        records: list[DataRecord] = []
        for row in dataframe.itertuples(index=False):
            records.append(
                DataRecord(
                    timestamp=row.date,
                    data_type=DataType.MARKET_BAR,
                    symbol=symbol,
                    payload={
                        "open": float(row.open),
                        "high": float(row.high),
                        "low": float(row.low),
                        "close": float(row.close),
                        "volume": int(row.volume),
                        "wap": float(row.wap),
                        "count": int(row.count),
                    },
                    source_id=self.source_id,
                    frequency=DataFrequency.SECOND_5,
                )
            )

        for record in records:
            if not self.validate_record(record):
                raise DataValidationError("Generated invalid DataRecord from CSV row")

        return records
