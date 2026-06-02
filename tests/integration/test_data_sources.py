"""Integration tests for pipeline data source implementations."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo

import pytest
from algotrading.api.schemas.backtesting import (
    BacktestRequest,
    build_backtest_data_request,
)
from algotrading.api.schemas.data_sources import (
    CachePolicy,
    MarketDataProvider,
    NewsDataProvider,
    normalize_training_data_request,
)
from algotrading.src.broker import BarData, InteractiveBrokersAdapter
from algotrading.src.data_pipeline.sources import BrokerDataSource, FileSource
from algotrading.src.data_pipeline.sources.exceptions import DataValidationError
from algotrading.src.data_pipeline.storage import (
    LocalDataCacheMissError,
    LocalDataStore,
)
from algotrading.src.data_pipeline.storage.local_store import MARKET_COLUMNS
from algotrading.src.data_pipeline.types import (
    DataBatch,
    DataFrequency,
    DataRecord,
    DataType,
    NewsRecord,
)

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


def test_canonical_data_request_and_local_store_round_trip(tmp_path: Path) -> None:
    """Normalize data-source configs and round-trip canonical local data."""

    start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    end = start + timedelta(minutes=1)
    raw_path = tmp_path / "raw" / "aapl.csv"
    model_data_config: dict[str, object] = {
        "symbols": [" aapl "],
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "data_frequency": "1m",
        "market": {
            "provider": "file",
            "symbol_files": {"aapl": str(raw_path)},
        },
        "news": {
            "enabled": True,
            "provider": "mock",
            "include_body": True,
            "limit": 25,
        },
    }

    request = normalize_training_data_request(
        model_data_config,
        {"cache_policy": "refresh"},
    )

    assert request.symbols == ["AAPL"]
    assert request.frequency == DataFrequency.MINUTE_1
    assert request.cache_policy == CachePolicy.REFRESH
    assert request.market_source.provider == MarketDataProvider.FILE
    assert request.market_source.explicit_files == {"AAPL": str(raw_path)}
    assert request.news_source.provider == NewsDataProvider.MOCK
    assert request.news_source.enabled is True

    backtest_request = BacktestRequest(
        model_id="model-1",
        generation_id="generation-1",
        start_date=date(2024, 1, 5),
        end_date=date(2024, 1, 6),
        symbols=["msft"],
    )
    backtest_data_request = build_backtest_data_request(
        backtest_request,
        model_data_config,
    )

    assert backtest_data_request.symbols == ["MSFT"]
    assert backtest_data_request.start_time.date() == backtest_request.start_date
    assert backtest_data_request.end_time.date() == backtest_request.end_date
    assert backtest_data_request.market_source.provider == MarketDataProvider.FILE
    assert backtest_data_request.news_source.provider == NewsDataProvider.MOCK

    store = LocalDataStore(root=tmp_path / "sourced")
    batch = DataBatch(
        records=[
            DataRecord(
                timestamp=start,
                data_type=DataType.MARKET_BAR,
                symbol="AAPL",
                payload={
                    "open": 190.0,
                    "high": 191.5,
                    "low": 189.5,
                    "close": 191.0,
                    "volume": 1000,
                    "vwap": 190.75,
                    "trade_count": 12,
                },
                source_id="file:aapl",
                frequency=DataFrequency.MINUTE_1,
            ),
            DataRecord(
                timestamp=end,
                data_type=DataType.MARKET_BAR,
                symbol="AAPL",
                payload={
                    "open": 191.0,
                    "high": 192.0,
                    "low": 190.8,
                    "close": 191.6,
                    "volume": 1200,
                    "vwap": 191.4,
                    "trade_count": 15,
                },
                source_id="file:aapl",
                frequency=DataFrequency.MINUTE_1,
            ),
        ],
        start_time=start,
        end_time=end,
        data_type=DataType.MARKET_BAR,
        symbol="AAPL",
    )

    market_manifest = store.write_market_batch(
        batch,
        provider=request.market_source.provider,
        frequency=request.frequency,
        requested_start=request.start_time,
        requested_end=request.end_time,
        cache_policy=request.cache_policy,
        source_file_paths=[raw_path],
    )
    market_path = tmp_path / "sourced" / "market" / "file" / "AAPL" / "MINUTE_1"

    assert market_manifest.provider == "file"
    assert market_manifest.record_count == 2
    assert (market_path / "manifest.json").exists()
    assert (market_path / "market.csv").read_text(encoding="utf-8").splitlines()[
        0
    ].split(",") == list(MARKET_COLUMNS)

    loaded_market = store.find_market_data(request)

    assert loaded_market["AAPL"].manifest.record_count == 2
    assert loaded_market["AAPL"].batch.records[-1].payload["close"] == 191.6
    assert loaded_market["AAPL"].data_path == market_path / "market.csv"

    news_record = NewsRecord(
        timestamp=start + timedelta(seconds=30),
        headline="Apple shares move after product update",
        body="Apple shares traded higher after a product update.",
        source="mock",
        symbols=["AAPL"],
        categories=["equities"],
        sentiment_score=0.35,
        url="https://example.test/aapl",
        news_id="mock-aapl-1",
    )
    news_manifest = store.write_news_records(
        [news_record],
        provider=request.news_source.provider,
        symbol="aapl",
        requested_start=request.start_time,
        requested_end=request.end_time,
        cache_policy=request.cache_policy,
    )
    news_path = tmp_path / "sourced" / "news" / "mock" / "AAPL"

    assert news_manifest.provider == "mock"
    assert news_manifest.record_count == 1
    assert (news_path / "manifest.json").exists()
    assert (news_path / "news.jsonl").exists()

    loaded_news = store.find_news_data(request)

    assert loaded_news["AAPL"].records[0].headline == news_record.headline
    assert loaded_news["AAPL"].data_path == news_path / "news.jsonl"

    outside_request = normalize_training_data_request(
        model_data_config,
        {"end_time": (end + timedelta(minutes=5)).isoformat()},
    )
    with pytest.raises(LocalDataCacheMissError, match="does not cover"):
        store.find_market_data(outside_request)


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
