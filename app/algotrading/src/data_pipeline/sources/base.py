"""Abstract interface for data sources used by the data pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Iterator

from algotrading.src.data_pipeline.types import (
    DataBatch,
    DataRecord,
    DataType,
    SourceMetadata,
)


class DataSource(ABC):
    """Abstract contract for pipeline data sources.

    Implementations provide connection lifecycle, batch retrieval, and stream
    retrieval methods for one or more symbols.
    """

    @property
    @abstractmethod
    def source_id(self) -> str:
        """Return unique source identifier."""

    @property
    @abstractmethod
    def metadata(self) -> SourceMetadata:
        """Return source capability metadata."""

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Return current connection state."""

    @abstractmethod
    def connect(self) -> None:
        """Initialize source resources and become ready for retrieval."""

    @abstractmethod
    def disconnect(self) -> None:
        """Release source resources and stop retrieval operations."""

    @abstractmethod
    def fetch_batch(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        data_type: DataType = DataType.MARKET_BAR,
    ) -> DataBatch:
        """Return all records for ``symbol`` in the requested time range."""

    @abstractmethod
    def fetch_stream(
        self,
        symbol: str,
        data_type: DataType = DataType.MARKET_BAR,
    ) -> Iterator[DataRecord]:
        """Yield records for ``symbol`` sequentially."""

    @abstractmethod
    def get_available_symbols(self) -> list[str]:
        """Return symbols that can be loaded from this source."""

    @abstractmethod
    def get_available_dates(self, symbol: str) -> list[date]:
        """Return available dates for ``symbol``."""

    def validate_record(self, record: DataRecord) -> bool:
        """Perform baseline record validation.

        Args:
            record: Record to validate.

        Returns:
            True when record satisfies baseline structural checks.
        """

        if not isinstance(record, DataRecord):
            return False
        if not isinstance(record.timestamp, datetime):
            return False
        if not isinstance(record.data_type, DataType):
            return False
        if not isinstance(record.payload, dict):
            return False
        if not record.symbol.strip():
            return False
        return True

    def __enter__(self) -> DataSource:
        """Connect source when entering context manager."""

        self.connect()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Disconnect source when exiting context manager."""

        self.disconnect()
