"""Integration tests for pipeline data source implementations."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo

import pytest
from algotrading.src.broker import BarData, InteractiveBrokersAdapter
from algotrading.src.data_pipeline.sources import BrokerDataSource, FileSource
from algotrading.src.data_pipeline.sources.exceptions import DataValidationError
from algotrading.src.data_pipeline.types import DataRecord, DataType

from tests.mocks import MockBrokerAdapter


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _saved_data_path() -> Path:
    return _repo_root() / "data" / "saved_data" / "pipeline_0001"


def test_file_source_batch_load() -> None:
    """Load historical market data as a single batch from FileSource."""

    source = FileSource(base_path=str(_saved_data_path()), trim_percentage=0.038)
    source.connect()

    eastern = ZoneInfo("US/Eastern")
    batch = source.fetch_batch(
        symbol="AMD",
        start=datetime(2022, 1, 3, 9, 30, tzinfo=eastern),
        end=datetime(2022, 1, 3, 16, 0, tzinfo=eastern),
    )

    assert len(batch.records) > 0
    assert batch.symbol == "AMD"
    assert batch.data_type == DataType.MARKET_BAR


def test_file_source_stream_mode() -> None:
    """Iterate file-backed records in stream mode."""

    source = FileSource(base_path=str(_saved_data_path()), trim_percentage=0.038)
    source.connect()

    stream = source.fetch_stream(symbol="AMD")
    first_five = [next(stream) for _ in range(5)]

    assert len(first_five) == 5
    assert all(record.symbol == "AMD" for record in first_five)
    assert all(record.data_type == DataType.MARKET_BAR for record in first_five)


def test_file_source_date_filtering() -> None:
    """Load only records from the requested date window."""

    source = FileSource(base_path=str(_saved_data_path()), trim_percentage=0.038)
    source.connect()

    eastern = ZoneInfo("US/Eastern")
    batch = source.fetch_batch(
        symbol="AMD",
        start=datetime(2022, 1, 4, 0, 0, tzinfo=eastern),
        end=datetime(2022, 1, 4, 23, 59, tzinfo=eastern),
    )

    assert len(batch.records) > 0
    assert {record.timestamp.date() for record in batch.records} == {
        datetime(2022, 1, 4, tzinfo=eastern).date()
    }


def test_file_source_validation_errors(tmp_path: Path) -> None:
    """Raise validation errors for malformed CSV input."""

    malformed = tmp_path / "AMD_NASDAQ_20240102.csv"
    malformed.write_text(
        "date,open,high,low,close,volume,count\n"
        "20240102 09:30:00 US/Eastern,100,101,99,100.5,1000,12\n",
        encoding="utf-8",
    )

    source = FileSource(base_path=str(tmp_path))
    source.connect()

    with pytest.raises(DataValidationError, match="Missing required columns"):
        source.fetch_batch(
            symbol="AMD",
            start=datetime(2024, 1, 2, 9, 30, tzinfo=ZoneInfo("US/Eastern")),
            end=datetime(2024, 1, 2, 16, 0, tzinfo=ZoneInfo("US/Eastern")),
        )


@pytest.mark.requires_ib
def test_broker_source_historical(confirm_ib_gateway: dict[str, object]) -> None:
    """Fetch historical bars through BrokerDataSource with a live IB adapter."""

    conn = confirm_ib_gateway
    adapter = InteractiveBrokersAdapter()
    source = BrokerDataSource(adapter=adapter)

    try:
        adapter.connect(
            str(conn["host"]),
            int(conn["port"]),
            int(conn["client_id"]) + 30,
        )

        end = datetime.now(tz=UTC)
        start = end - timedelta(minutes=15)
        batch = source.fetch_batch(symbol="AMD", start=start, end=end)

        assert len(batch.records) > 0
        assert batch.records[-1].payload["close"] > 0
    finally:
        adapter.disconnect()


@pytest.mark.requires_ib
def test_broker_source_streaming(confirm_ib_gateway: dict[str, object]) -> None:
    """Consume at least one live realtime record through BrokerDataSource."""

    conn = confirm_ib_gateway
    adapter = InteractiveBrokersAdapter()
    source = BrokerDataSource(adapter=adapter)

    try:
        adapter.connect(
            str(conn["host"]),
            int(conn["port"]),
            int(conn["client_id"]) + 31,
        )
        iterator = source.fetch_stream(symbol="AMD")

        result: dict[str, DataRecord] = {}
        errors: dict[str, Exception] = {}
        ready = threading.Event()

        def _consume_one() -> None:
            try:
                result["record"] = next(iterator)
            except Exception as exc:  # pragma: no cover - defensive in thread
                errors["error"] = exc
            finally:
                ready.set()

        worker = threading.Thread(target=_consume_one, daemon=True)
        worker.start()
        completed = ready.wait(timeout=25.0)

        adapter.disconnect()
        worker.join(timeout=5.0)

        if not completed:
            pytest.skip("No realtime bars received within timeout window.")
        if "error" in errors:
            raise errors["error"]

        record = result["record"]
        assert record.symbol == "AMD"
        assert record.data_type == DataType.MARKET_BAR
        assert "close" in record.payload
    finally:
        if adapter.is_connected():
            adapter.disconnect()


class _SourceLike(Protocol):
    """Common test protocol for source interchangeability checks."""

    source_id: str

    def connect(self) -> None:
        """Connect source for retrieval."""

    def fetch_batch(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        data_type: DataType = DataType.MARKET_BAR,
    ) -> object:
        """Fetch a batch from the source."""


def test_source_interchangeability() -> None:
    """Verify different source implementations satisfy the same batch contract."""

    broker_start = datetime(2022, 1, 3, 14, 30, tzinfo=UTC)
    broker_end = broker_start + timedelta(seconds=15)
    eastern = ZoneInfo("US/Eastern")
    file_start = datetime(2022, 1, 3, 0, 0, tzinfo=eastern)
    file_end = datetime(2022, 1, 3, 23, 59, tzinfo=eastern)

    file_source = FileSource(base_path=str(_saved_data_path()), trim_percentage=0.038)

    mock_adapter = MockBrokerAdapter()
    mock_adapter.connect("127.0.0.1", 7497, 901)
    mock_adapter.set_historical_data(
        [
            BarData(
                timestamp=broker_start,
                open=100.0,
                high=100.5,
                low=99.8,
                close=100.2,
                volume=1000,
                vwap=100.1,
                trade_count=10,
            ),
            BarData(
                timestamp=broker_start + timedelta(seconds=5),
                open=100.2,
                high=100.6,
                low=100.0,
                close=100.4,
                volume=1100,
                vwap=100.3,
                trade_count=11,
            ),
        ]
    )
    broker_source = BrokerDataSource(adapter=mock_adapter)

    sources: list[tuple[_SourceLike, str]] = [
        (file_source, "AMD"),
        (broker_source, "AMD"),
    ]

    for source, symbol in sources:
        source.connect()
        if isinstance(source, FileSource):
            batch = source.fetch_batch(
                symbol=symbol,
                start=file_start,
                end=file_end,
            )
        else:
            batch = source.fetch_batch(
                symbol=symbol, start=broker_start, end=broker_end
            )
        assert len(batch.records) > 0
        assert batch.symbol == symbol
