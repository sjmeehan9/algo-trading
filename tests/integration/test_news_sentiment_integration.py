"""Integration tests for news sentiment supporting model registration and inference."""

from __future__ import annotations

import time
from datetime import UTC, date, datetime
from typing import Iterator

from algotrading.src.data_pipeline import (
    DataBatch,
    DataFrequency,
    DataRecord,
    DataType,
    SourceMetadata,
)
from algotrading.src.data_pipeline.routing import DataRouter, RouteConfig, RouterConfig
from algotrading.src.data_pipeline.sources.base import DataSource
from algotrading.src.data_pipeline.sources.mock_news_source import MockNewsSource
from algotrading.src.models.inference import InferencePipeline
from algotrading.src.models.registry import (
    CustomStrategyRegistry,
    ModelEntryConfig,
    ModelState,
    SupportingModelRegistry,
)
from algotrading.src.models.signals import (
    SignalType,
    validate_signal,
    validate_signal_for_type,
)


class _SingleNewsRecordSource(DataSource):
    """Deterministic source that emits one news record and exits."""

    def __init__(self, record: DataRecord) -> None:
        self._record = record
        self._connected = False

    @property
    def source_id(self) -> str:
        return "single-news-source"

    @property
    def metadata(self) -> SourceMetadata:
        return SourceMetadata(
            source_id=self.source_id,
            source_type="memory",
            supported_types=[self._record.data_type],
            supported_frequencies=[self._record.frequency or DataFrequency.IRREGULAR],
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
        if (
            symbol == self._record.symbol
            and data_type == self._record.data_type
            and start <= self._record.timestamp <= end
        ):
            records = [self._record]
        else:
            records = []

        return DataBatch(
            records=records,
            start_time=start,
            end_time=end,
            data_type=data_type,
            symbol=symbol,
        )

    def fetch_stream(
        self,
        symbol: str,
        data_type: DataType = DataType.MARKET_BAR,
    ) -> Iterator[DataRecord]:
        del symbol, data_type
        yield self._record

    def get_available_symbols(self) -> list[str]:
        return [self._record.symbol]

    def get_available_dates(self, symbol: str) -> list[date]:
        del symbol
        return [self._record.timestamp.date()]


def _wait_until(predicate, timeout_seconds: float = 3.0) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_news_sentiment_trainer_registers_and_loads() -> None:
    registry = SupportingModelRegistry()
    model_id = "news-sentiment-vader"
    registry.register(
        ModelEntryConfig(
            model_id=model_id,
            model_type="ml",
            signal_type=SignalType.SENTIMENT,
            trainer_class="algotrading.src.models.supporting.sentiment.news_sentiment.NewsSentimentTrainer",
            input_data_types=[DataType.NEWS_TEXT],
            input_frequency=DataFrequency.IRREGULAR,
        )
    )

    registry.load_model(model_id)
    registry.set_state(model_id, ModelState.READY)

    entry = registry.get(model_id)
    assert entry is not None
    assert entry.trainer is not None
    assert entry.state == ModelState.READY


def test_news_sentiment_inference_pipeline_produces_valid_signal(tmp_path) -> None:
    record = DataRecord(
        timestamp=datetime.now(tz=UTC),
        data_type=DataType.NEWS_TEXT,
        symbol="AAPL",
        payload={
            "headline": "Strong subscription growth boosts outlook",
            "body": "Management raised revenue guidance for the next quarter.",
        },
        frequency=DataFrequency.IRREGULAR,
        source_id="news-source",
    )

    source = _SingleNewsRecordSource(record)
    router = DataRouter(
        RouterConfig(
            routes=[
                RouteConfig(
                    source=source,
                    data_types=[DataType.NEWS_TEXT],
                    symbols=["AAPL"],
                    targets=["callback"],
                )
            ]
        )
    )

    model_registry = SupportingModelRegistry()
    model_id = "news-sentiment-pipeline"
    model_registry.register(
        ModelEntryConfig(
            model_id=model_id,
            model_type="ml",
            signal_type=SignalType.SENTIMENT,
            trainer_class="algotrading.src.models.supporting.sentiment.news_sentiment.NewsSentimentTrainer",
            input_data_types=[DataType.NEWS_TEXT],
            input_frequency=DataFrequency.IRREGULAR,
        )
    )
    model_registry.load_model(model_id)
    model_registry.set_state(model_id, ModelState.READY)

    strategy_dir = tmp_path / "strategies"
    strategy_dir.mkdir(parents=True, exist_ok=True)
    strategy_registry = CustomStrategyRegistry(
        strategy_dirs=[str(strategy_dir)],
        auto_scan=True,
    )

    pipeline = InferencePipeline(
        model_registry=model_registry,
        strategy_registry=strategy_registry,
        data_router=router,
    )
    pipeline.configure(
        {
            "max_workers": 2,
            "default_timeout_seconds": 2.0,
            "prune_every_n_results": 100,
        }
    )

    pipeline.start()
    router.start()

    assert _wait_until(
        lambda: model_id in pipeline.get_all_latest_signals(),
        timeout_seconds=5.0,
    )

    signal = pipeline.get_latest_signal(model_id)
    assert signal is not None
    assert signal.signal_type == SignalType.SENTIMENT
    is_valid, _ = validate_signal(signal)
    assert is_valid
    is_type_valid, _ = validate_signal_for_type(signal)
    assert is_type_valid

    metrics = pipeline.get_metrics()
    assert metrics.total_inferences >= 1
    assert metrics.failed_inferences == 0

    router.stop()
    pipeline.stop()


def test_news_sentiment_direct_subscription_flow(tmp_path) -> None:
    """Configured news source subscription should feed sentiment inference."""

    router = DataRouter(RouterConfig(routes=[]))

    model_registry = SupportingModelRegistry()
    model_id = "news-sentiment-direct"
    model_registry.register(
        ModelEntryConfig(
            model_id=model_id,
            model_type="ml",
            signal_type=SignalType.SENTIMENT,
            trainer_class="algotrading.src.models.supporting.sentiment.news_sentiment.NewsSentimentTrainer",
            input_data_types=[DataType.NEWS_TEXT],
            input_frequency=DataFrequency.IRREGULAR,
        )
    )
    model_registry.load_model(model_id)
    model_registry.set_state(model_id, ModelState.READY)

    strategy_dir = tmp_path / "strategies"
    strategy_dir.mkdir(parents=True, exist_ok=True)
    strategy_registry = CustomStrategyRegistry(
        strategy_dirs=[str(strategy_dir)],
        auto_scan=True,
    )

    news_source = MockNewsSource(symbols=["AAPL"], frequency_seconds=1)

    pipeline = InferencePipeline(
        model_registry=model_registry,
        strategy_registry=strategy_registry,
        data_router=router,
    )
    pipeline.configure(
        {
            "max_workers": 2,
            "default_timeout_seconds": 2.0,
            "prune_every_n_results": 100,
        }
    )
    pipeline.configure_news_sentiment(
        news_source=news_source,
        sentiment_model_id=model_id,
        symbols=["AAPL"],
    )

    pipeline.start()

    assert _wait_until(
        lambda: pipeline.get_latest_signal(model_id) is not None,
        timeout_seconds=5.0,
    )

    signal = pipeline.get_latest_signal(model_id)
    assert signal is not None
    assert signal.signal_type == SignalType.SENTIMENT
    assert signal.symbol == "AAPL"

    pipeline.stop()
