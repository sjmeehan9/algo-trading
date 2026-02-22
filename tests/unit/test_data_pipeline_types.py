"""Unit tests for data pipeline foundational types."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from algotrading.src.data_pipeline import (
    DataBatch,
    DataFrequency,
    DataRecord,
    DataType,
    SourceMetadata,
    __version__,
)


def _make_record(
    timestamp: datetime,
    *,
    symbol: str = "AAPL",
    data_type: DataType = DataType.MARKET_BAR,
    close: float = 100.0,
) -> DataRecord:
    """Create a default valid record for tests."""

    return DataRecord(
        timestamp=timestamp,
        data_type=data_type,
        symbol=symbol,
        payload={"close": close, "volume": 1_000},
        source_id="test_source",
        frequency=DataFrequency.MINUTE_1,
    )


def test_data_record_is_immutable() -> None:
    """DataRecord is frozen and rejects mutation."""

    record = _make_record(datetime(2026, 2, 23, 14, 30, tzinfo=UTC))

    with pytest.raises(FrozenInstanceError):
        record.symbol = "MSFT"  # type: ignore[misc]


def test_data_record_requires_non_empty_symbol() -> None:
    """DataRecord validates symbol as non-empty."""

    with pytest.raises(ValueError, match="symbol must be non-empty"):
        DataRecord(
            timestamp=datetime(2026, 2, 23, 14, 30, tzinfo=UTC),
            data_type=DataType.MARKET_BAR,
            symbol="   ",
            payload={"close": 100.0},
        )


def test_data_batch_to_dataframe_flattens_record_payload() -> None:
    """DataBatch.to_dataframe returns expected flattened schema."""

    t0 = datetime(2026, 2, 23, 14, 30, tzinfo=UTC)
    records = [
        _make_record(t0, close=100.0),
        _make_record(t0 + timedelta(minutes=1), close=101.0),
    ]
    batch = DataBatch(
        records=records,
        start_time=t0,
        end_time=t0 + timedelta(minutes=1),
        data_type=DataType.MARKET_BAR,
        symbol="AAPL",
    )

    df = batch.to_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert {
        "timestamp",
        "data_type",
        "symbol",
        "source_id",
        "frequency",
        "close",
        "volume",
    }.issubset(df.columns)
    assert df["close"].tolist() == [100.0, 101.0]
    assert df["data_type"].tolist() == ["MARKET_BAR", "MARKET_BAR"]


def test_data_batch_filter_by_time_returns_subset() -> None:
    """DataBatch.filter_by_time returns a narrowed batch window and records."""

    t0 = datetime(2026, 2, 23, 14, 30, tzinfo=UTC)
    records = [
        _make_record(t0, close=100.0),
        _make_record(t0 + timedelta(minutes=1), close=101.0),
        _make_record(t0 + timedelta(minutes=2), close=102.0),
    ]
    batch = DataBatch(
        records=records,
        start_time=t0,
        end_time=t0 + timedelta(minutes=2),
        data_type=DataType.MARKET_BAR,
        symbol="AAPL",
    )

    filtered = batch.filter_by_time(
        t0 + timedelta(minutes=1), t0 + timedelta(minutes=2)
    )

    assert isinstance(filtered, DataBatch)
    assert len(filtered) == 2
    assert [r.payload["close"] for r in filtered] == [101.0, 102.0]
    assert filtered.start_time == t0 + timedelta(minutes=1)
    assert filtered.end_time == t0 + timedelta(minutes=2)


def test_data_batch_supports_len_and_iteration() -> None:
    """DataBatch supports collection-like length and iteration semantics."""

    t0 = datetime(2026, 2, 23, 14, 30, tzinfo=UTC)
    batch = DataBatch(
        records=[_make_record(t0), _make_record(t0 + timedelta(minutes=1))],
        start_time=t0,
        end_time=t0 + timedelta(minutes=1),
        data_type=DataType.MARKET_BAR,
        symbol="AAPL",
    )

    assert len(batch) == 2
    assert all(isinstance(record, DataRecord) for record in batch)


def test_enum_values_match_component_spec() -> None:
    """Enum values align with Phase 3.1 type definitions."""

    assert DataType.MARKET_BAR.value == "MARKET_BAR"
    assert DataType.NEWS_TEXT.value == "NEWS_TEXT"
    assert DataType.INDICATOR.value == "INDICATOR"
    assert DataType.SIGNAL.value == "SIGNAL"

    assert DataFrequency.TICK.value == "TICK"
    assert DataFrequency.SECOND_5.value == "SECOND_5"
    assert DataFrequency.MINUTE_1.value == "MINUTE_1"
    assert DataFrequency.IRREGULAR.value == "IRREGULAR"


def test_public_import_surface_and_source_metadata_creation() -> None:
    """Public package imports work and metadata object is constructible."""

    metadata = SourceMetadata(
        source_id="file_source_1",
        source_type="file",
        supported_types=[DataType.MARKET_BAR],
        supported_frequencies=[DataFrequency.MINUTE_1],
        config={"path": "./data"},
    )

    assert metadata.source_id == "file_source_1"
    assert __version__ == "1.0.0"
