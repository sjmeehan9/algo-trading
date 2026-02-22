"""Unit tests for the DataSource abstract interface."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Iterator

import pytest
from algotrading.src.data_pipeline.sources.base import DataSource
from algotrading.src.data_pipeline.types import (
    DataBatch,
    DataRecord,
    DataType,
    SourceMetadata,
)


def test_data_source_is_abstract() -> None:
    """DataSource cannot be instantiated directly."""

    with pytest.raises(TypeError):
        DataSource()


class _ConcreteSource(DataSource):
    """Minimal concrete source implementation for testing base behavior."""

    def __init__(self) -> None:
        self._connected = False

    @property
    def source_id(self) -> str:
        return "test_source"

    @property
    def metadata(self) -> SourceMetadata:
        return SourceMetadata(
            source_id=self.source_id,
            source_type="test",
            supported_types=[DataType.MARKET_BAR],
            supported_frequencies=[],
            config={},
        )

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def fetch_batch(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        data_type: DataType = DataType.MARKET_BAR,
    ) -> DataBatch:
        del symbol, start, end, data_type
        return DataBatch(
            records=[],
            start_time=datetime(2026, 2, 23, 14, 30, tzinfo=UTC),
            end_time=datetime(2026, 2, 23, 14, 30, tzinfo=UTC),
            data_type=DataType.MARKET_BAR,
            symbol="AMD",
        )

    def fetch_stream(
        self,
        symbol: str,
        data_type: DataType = DataType.MARKET_BAR,
    ) -> Iterator[DataRecord]:
        del symbol, data_type
        return iter([])

    def get_available_symbols(self) -> list[str]:
        return ["AMD"]

    def get_available_dates(self, symbol: str) -> list[date]:
        del symbol
        return [date(2026, 2, 23)]


def test_validate_record_accepts_valid_record() -> None:
    """validate_record returns True for valid records."""

    source = _ConcreteSource()
    record = DataRecord(
        timestamp=datetime(2026, 2, 23, 14, 30, tzinfo=UTC),
        data_type=DataType.MARKET_BAR,
        symbol="AMD",
        payload={"close": 120.0},
    )

    assert source.validate_record(record) is True


def test_context_manager_connects_and_disconnects() -> None:
    """DataSource context manager calls connect and disconnect."""

    source = _ConcreteSource()
    assert source.is_connected is False

    with source as connected:
        assert connected.is_connected is True

    assert source.is_connected is False
