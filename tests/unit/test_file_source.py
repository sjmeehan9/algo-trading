"""Unit tests for FileSource behavior and CSV parsing."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from algotrading.src.data_pipeline.sources.exceptions import (
    DataNotFoundError,
    DataSourceConnectionError,
    DataValidationError,
)
from algotrading.src.data_pipeline.sources.file_source import FileSource


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _sample_csv_legacy_timezone() -> str:
    return (
        "date,open,high,low,close,volume,wap,count\n"
        "20220103 10:00:00 US/Eastern,148.0,148.2,147.9,148.1,1000,148.05,10\n"
        "20220103 10:00:05 US/Eastern,148.1,148.3,148.0,148.2,1100,148.15,11\n"
        "20220103 10:00:10 US/Eastern,148.2,148.4,148.1,148.3,1200,148.25,12\n"
        "20220103 10:00:15 US/Eastern,148.3,148.5,148.2,148.4,1300,148.35,13\n"
    )


def test_file_source_requires_connect_before_fetch(tmp_path: Path) -> None:
    """FileSource rejects fetch calls before connect()."""

    source = FileSource(base_path=str(tmp_path))
    with pytest.raises(DataSourceConnectionError):
        source.fetch_stream(symbol="AMD")


def test_fetch_batch_filters_requested_time_range(tmp_path: Path) -> None:
    """fetch_batch returns records constrained to start/end datetimes."""

    _write_file(tmp_path / "AMD_NASDAQ_20220103.csv", _sample_csv_legacy_timezone())

    source = FileSource(base_path=str(tmp_path), trim_percentage=0.0)
    source.connect()

    eastern = ZoneInfo("US/Eastern")
    batch = source.fetch_batch(
        symbol="AMD",
        start=datetime(2022, 1, 3, 10, 0, 5, tzinfo=eastern),
        end=datetime(2022, 1, 3, 10, 0, 10, tzinfo=eastern),
    )

    assert len(batch.records) == 2
    assert batch.records[0].payload["close"] == 148.2
    assert batch.records[1].payload["close"] == 148.3


def test_fetch_stream_yields_records_in_order(tmp_path: Path) -> None:
    """fetch_stream yields one DataRecord at a time in timestamp order."""

    _write_file(tmp_path / "AMD_NASDAQ_20220103.csv", _sample_csv_legacy_timezone())
    source = FileSource(base_path=str(tmp_path), trim_percentage=0.0)
    source.connect()

    records = list(source.fetch_stream(symbol="AMD"))

    assert len(records) == 4
    assert records[0].timestamp <= records[-1].timestamp
    assert records[0].payload["open"] == 148.0


def test_fetch_batch_raises_when_symbol_not_found(tmp_path: Path) -> None:
    """Unknown symbols raise DataNotFoundError."""

    _write_file(tmp_path / "AMD_NASDAQ_20220103.csv", _sample_csv_legacy_timezone())
    source = FileSource(base_path=str(tmp_path), trim_percentage=0.0)
    source.connect()

    with pytest.raises(DataNotFoundError):
        source.fetch_batch(
            symbol="MSFT",
            start=datetime(2022, 1, 3, 0, 0, tzinfo=ZoneInfo("US/Eastern")),
            end=datetime(2022, 1, 3, 23, 59, tzinfo=ZoneInfo("US/Eastern")),
        )


def test_validation_error_for_missing_required_columns(tmp_path: Path) -> None:
    """Missing required CSV columns raise DataValidationError."""

    invalid_csv = (
        "date,open,high,low,close,volume,wap\n"
        "20220103 10:00:00 US/Eastern,148.0,148.2,147.9,148.1,1000,148.05\n"
    )
    _write_file(tmp_path / "AMD_NASDAQ_20220103.csv", invalid_csv)

    source = FileSource(base_path=str(tmp_path), trim_percentage=0.0)
    source.connect()

    with pytest.raises(DataValidationError, match="Missing required columns"):
        source.fetch_batch(
            symbol="AMD",
            start=datetime(2022, 1, 3, 0, 0, tzinfo=ZoneInfo("US/Eastern")),
            end=datetime(2022, 1, 3, 23, 59, tzinfo=ZoneInfo("US/Eastern")),
        )
