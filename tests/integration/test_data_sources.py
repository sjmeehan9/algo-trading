"""Integration tests for pipeline data source implementations."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo

import pytest
from algotrading.api.config import APIConfig
from algotrading.api.main import create_app
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
from algotrading.api.services.data_acquisition_service import DataAcquisitionService
from algotrading.src.broker import BarData, InteractiveBrokersAdapter
from algotrading.src.broker.registry import BrokerRegistry
from algotrading.src.data_pipeline.acquisition import (
    AcquisitionStatus,
    HistoricalMarketDataAcquirer,
    HistoricalNewsAcquirer,
    NewsDataAcquisitionError,
    model_requires_news,
)
from algotrading.src.data_pipeline.sources import BrokerDataSource, FileSource
from algotrading.src.data_pipeline.sources.exceptions import DataValidationError
from algotrading.src.data_pipeline.sources.news_factory import NewsSourceFactory
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
from algotrading.src.models.registry import ModelEntryConfig, SupportingModelRegistry
from algotrading.src.models.signals import SignalType
from fastapi.testclient import TestClient

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

    first_bar_time = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    start = first_bar_time - timedelta(seconds=30)
    end = first_bar_time + timedelta(minutes=1, seconds=30)
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


def test_historical_market_acquirer_file_backed_writes_canonical_store(
    tmp_path: Path,
) -> None:
    """Acquire explicit CSV market data and persist canonical local artifacts."""

    start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    end = start + timedelta(minutes=1)
    raw_path = tmp_path / "raw" / "aapl.csv"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text(
        "timestamp,symbol,open,high,low,close,volume,vwap,trade_count\n"
        "2024-01-02T14:30:00+00:00,AAPL,190,191,189.5,190.5,1000,190.2,10\n"
        "2024-01-02T14:31:00+00:00,AAPL,190.5,192,190,191.7,1200,191.2,12\n",
        encoding="utf-8",
    )
    request = normalize_training_data_request(
        {
            "symbols": ["aapl"],
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "data_frequency": "1m",
            "market": {
                "provider": "file",
                "symbol_files": {"aapl": str(raw_path)},
            },
        },
        {"cache_policy": "refresh"},
    )
    store = LocalDataStore(root=tmp_path / "sourced")
    acquirer = HistoricalMarketDataAcquirer(store=store)

    result = acquirer.acquire(request)

    assert result.provider == MarketDataProvider.FILE.value
    assert result.reports[0].status == AcquisitionStatus.ACQUIRED.value
    assert result.reports[0].row_count == 2
    assert Path(result.reports[0].data_path or "").exists()
    assert Path(result.reports[0].manifest_path or "").exists()

    cached_request = normalize_training_data_request(
        request.model_dump(mode="json"),
        {"cache_policy": CachePolicy.PREFER_CACHE.value},
    )
    cached_result = acquirer.acquire(cached_request)
    loaded = store.find_market_data(cached_request)

    assert cached_result.reports[0].status == AcquisitionStatus.CACHED.value
    assert loaded["AAPL"].batch.records[-1].payload["close"] == 191.7
    assert loaded["AAPL"].manifest.source_file_paths == (str(raw_path),)


def test_market_acquisition_api_accepts_loose_file_backed_payload(
    tmp_path: Path,
) -> None:
    """Acquire file-backed market data through the FastAPI route."""

    start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    end = start + timedelta(minutes=1)
    raw_path = tmp_path / "raw" / "msft.csv"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text(
        "date,symbol,open,high,low,close,volume,wap,count\n"
        "2024-01-02T14:30:00+00:00,MSFT,370,371,369,370.5,800,370.2,8\n"
        "2024-01-02T14:31:00+00:00,MSFT,370.5,372,370,371.2,900,371.0,9\n",
        encoding="utf-8",
    )
    api_config = APIConfig(
        api_key="data-secret-key",
        debug=True,
        cors_origins=["http://localhost:3000"],
    )
    app = create_app(api_config)
    store = LocalDataStore(root=tmp_path / "sourced")
    app.state.data_acquisition_service = DataAcquisitionService(
        store=store,
        api_config=api_config,
        broker_registry=BrokerRegistry.isolated(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/data/market/acquire",
            headers={"X-API-Key": "data-secret-key"},
            json={
                "symbols": ["msft"],
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "data_frequency": "1m",
                "cache_policy": "refresh",
                "market": {
                    "provider": "file",
                    "symbol_files": {"msft": str(raw_path)},
                },
            },
        )

    assert response.status_code == 200, response.text
    payload = response.json()["data"]
    assert payload["provider"] == "file"
    assert payload["reports"][0]["symbol"] == "MSFT"
    assert payload["reports"][0]["row_count"] == 2
    assert Path(payload["reports"][0]["data_path"]).exists()

    request = normalize_training_data_request(
        {
            "symbols": ["MSFT"],
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "data_frequency": "1m",
            "market": {"provider": "file"},
        }
    )
    assert store.find_market_data(request)["MSFT"].manifest.record_count == 2


def test_historical_news_acquirer_detects_required_news_and_persists_mock(
    tmp_path: Path,
) -> None:
    """Acquire required mock news for a model with NEWS_TEXT supporting inputs."""

    start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    end = start + timedelta(minutes=3)
    supporting_model_id = "sentiment-news-1"
    registry = SupportingModelRegistry()
    registry.register(
        ModelEntryConfig(
            model_id=supporting_model_id,
            model_type="ml",
            signal_type=SignalType.SENTIMENT,
            trainer_class="algotrading.src.models.supporting.sentiment.news_sentiment.NewsSentimentTrainer",
            input_data_types=[DataType.NEWS_TEXT],
            input_frequency=DataFrequency.IRREGULAR,
        )
    )
    model = {"supporting_model_ids": [supporting_model_id]}
    request = normalize_training_data_request(
        {
            "symbols": ["aapl"],
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "data_frequency": "1m",
            "cache_policy": "refresh",
            "market": {"provider": "file"},
            "news": {
                "enabled": True,
                "provider": "mock",
                "include_body": True,
                "limit": 10,
            },
        }
    )
    store = LocalDataStore(root=tmp_path / "sourced")
    acquirer = HistoricalNewsAcquirer(
        store=store,
        supporting_registry=registry,
        default_provider=NewsDataProvider.MOCK,
    )

    assert model_requires_news(model, registry) is True

    result = acquirer.acquire(request, model=model)
    loaded = store.find_news_data(request)

    assert result.provider == NewsDataProvider.MOCK.value
    assert result.frequency == DataFrequency.IRREGULAR.value
    assert result.reports[0].status == AcquisitionStatus.ACQUIRED.value
    assert result.reports[0].row_count > 0
    assert Path(result.reports[0].data_path or "").exists()
    assert Path(result.reports[0].manifest_path or "").exists()
    assert loaded["AAPL"].records[0].headline


def test_required_news_failure_blocks_acquisition(tmp_path: Path) -> None:
    """Required news should fail loudly when no cache or provider can satisfy it."""

    start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    end = start + timedelta(minutes=1)
    request = normalize_training_data_request(
        {
            "symbols": ["aapl"],
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "data_frequency": "1m",
            "cache_policy": "refresh",
            "market": {"provider": "file"},
            "news": {"enabled": True, "required": True, "provider": "none"},
        }
    )
    acquirer = HistoricalNewsAcquirer(store=LocalDataStore(root=tmp_path / "sourced"))

    with pytest.raises(NewsDataAcquisitionError, match="no news provider"):
        acquirer.acquire(request)


def test_news_acquisition_api_accepts_loose_mock_payload(tmp_path: Path) -> None:
    """Acquire mock news through the FastAPI route and canonical service path."""

    start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    end = start + timedelta(minutes=2)
    api_config = APIConfig(
        api_key="data-secret-key",
        debug=True,
        cors_origins=["http://localhost:3000"],
    )
    app = create_app(api_config)
    store = LocalDataStore(root=tmp_path / "sourced")
    app.state.data_acquisition_service = DataAcquisitionService(
        store=store,
        api_config=api_config,
        broker_registry=BrokerRegistry.isolated(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/data/news/acquire",
            headers={"X-API-Key": "data-secret-key"},
            json={
                "symbols": ["aapl"],
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "data_frequency": "1m",
                "cache_policy": "refresh",
                "market": {"provider": "file"},
                "news": {
                    "enabled": True,
                    "provider": "mock",
                    "include_body": True,
                    "limit": 5,
                },
            },
        )

    assert response.status_code == 200, response.text
    payload = response.json()["data"]
    assert payload["provider"] == "mock"
    assert payload["reports"][0]["symbol"] == "AAPL"
    assert payload["reports"][0]["row_count"] > 0
    assert Path(payload["reports"][0]["data_path"]).exists()

    request = normalize_training_data_request(
        {
            "symbols": ["AAPL"],
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "data_frequency": "1m",
            "market": {"provider": "file"},
            "news": {"enabled": True, "provider": "mock"},
        }
    )
    assert store.find_news_data(request)["AAPL"].manifest.record_count > 0


@pytest.mark.requires_news_api
@pytest.mark.slow
def test_news_acquirer_primary_provider_persists_canonical_store(
    confirm_news_api: dict[str, str],
    tmp_path: Path,
) -> None:
    """Acquire real provider news and persist canonical JSONL plus manifest."""

    end = datetime.now(tz=UTC)
    start = end - timedelta(days=7)
    request = normalize_training_data_request(
        {
            "symbols": ["AAPL"],
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "data_frequency": "1d",
            "cache_policy": "refresh",
            "market": {"provider": "file"},
            "news": {
                "enabled": True,
                "required": True,
                "provider": "benzinga",
                "fallback_provider": "alphavantage",
                "include_body": False,
                "limit": 10,
            },
        }
    )
    factory = NewsSourceFactory.from_files(
        providers_path=confirm_news_api["providers_path"],
        credentials_path=confirm_news_api["credentials_path"],
    )
    store = LocalDataStore(root=tmp_path / "sourced")
    acquirer = HistoricalNewsAcquirer(store=store, news_source_factory=factory)

    result = acquirer.acquire(request)
    loaded = store.find_news_data(request)

    assert result.reports[0].row_count > 0
    assert Path(result.reports[0].manifest_path or "").exists()
    assert loaded["AAPL"].records[0].news_id


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
def test_market_acquirer_ib_persists_canonical_store(
    confirm_ib_gateway: dict[str, object],
    tmp_path: Path,
) -> None:
    """Acquire IB historical bars and persist canonical market data."""

    conn = confirm_ib_gateway
    # Use a known regular-session window so the provider-confirm test remains
    # stable after market close and on weekends.
    start = datetime(2026, 2, 13, 15, 0, tzinfo=UTC)
    end = start + timedelta(minutes=5)
    request = normalize_training_data_request(
        {
            "symbols": ["AMD"],
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "data_frequency": "5s",
            "cache_policy": "refresh",
            "market": {
                "provider": "ib",
                "bar_size": "5 secs",
                "broker_data_type": "TRADES",
                "exchange": "SMART",
                "currency": "USD",
            },
        }
    )
    store = LocalDataStore(root=tmp_path / "sourced")
    acquirer = HistoricalMarketDataAcquirer(
        store=store,
        broker_registry=BrokerRegistry.isolated(),
        broker_config={},
        broker_connection_params={
            "host": str(conn["host"]),
            "port": int(conn["port"]),
            "client_id": int(conn["client_id"]) + 52,
        },
    )

    result = acquirer.acquire(request)
    loaded = store.find_market_data(request)

    assert result.provider == MarketDataProvider.IB.value
    assert result.reports[0].row_count > 0
    assert result.reports[0].provider == MarketDataProvider.IB.value
    assert Path(result.reports[0].data_path or "").exists()
    assert loaded["AMD"].manifest.record_count == result.reports[0].row_count


@pytest.mark.requires_alpaca
def test_market_acquirer_alpaca_persists_canonical_store(
    confirm_alpaca_paper: dict[str, str],
    tmp_path: Path,
) -> None:
    """Acquire Alpaca historical bars and persist canonical market data."""

    pytest.importorskip("alpaca")
    end = datetime.now(tz=UTC) - timedelta(minutes=20)
    start = end - timedelta(days=5)
    request = normalize_training_data_request(
        {
            "symbols": ["AAPL"],
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "data_frequency": "1m",
            "cache_policy": "refresh",
            "market": {
                "provider": "alpaca",
                "bar_size": "1 min",
                "broker_data_type": "TRADES",
                "exchange": "SMART",
                "currency": "USD",
                "primary_exchange": "NASDAQ",
            },
        }
    )
    store = LocalDataStore(root=tmp_path / "sourced")
    acquirer = HistoricalMarketDataAcquirer(
        store=store,
        broker_registry=BrokerRegistry.isolated(),
        broker_config={**confirm_alpaca_paper, "paper": True, "data_feed": "iex"},
        broker_connection_params={},
    )

    result = acquirer.acquire(request)
    loaded = store.find_market_data(request)

    assert result.provider == MarketDataProvider.ALPACA.value
    assert result.reports[0].row_count > 0
    assert result.reports[0].provider == MarketDataProvider.ALPACA.value
    assert Path(result.reports[0].manifest_path or "").exists()
    assert loaded["AAPL"].batch.records[-1].payload["close"] > 0


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
