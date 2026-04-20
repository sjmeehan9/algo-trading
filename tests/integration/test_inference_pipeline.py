"""Integration tests for asynchronous inference pipeline."""

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
from algotrading.src.models.inference import InferencePipeline
from algotrading.src.models.registry import (
    CustomStrategyRegistry,
    ModelEntryConfig,
    SupportingModelRegistry,
)
from algotrading.src.models.signals import ModelSignal, SignalType


class _SingleRecordSource(DataSource):
    """Deterministic source that emits one news record then ends stream."""

    def __init__(self, record: DataRecord) -> None:
        self._record = record
        self._connected = False

    @property
    def source_id(self) -> str:
        return "single-record-source"

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
            self._record.symbol == symbol
            and self._record.data_type == data_type
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


class _NewsTrainer:
    """Simple trainer converting text payload into numeric signal values."""

    def __init__(self, value: float, delay_seconds: float = 0.0) -> None:
        self._value = value
        self._delay_seconds = delay_seconds

    def predict(self, value: object) -> float:
        del value
        if self._delay_seconds > 0:
            time.sleep(self._delay_seconds)
        return self._value


def _wait_until(predicate, timeout_seconds: float = 3.0) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _build_pipeline(tmp_path, trainer_delays: tuple[float, float] = (0.0, 0.0)):
    record = DataRecord(
        timestamp=datetime.now(tz=UTC),
        data_type=DataType.NEWS_TEXT,
        symbol="AAPL",
        payload={"headline": "Strong earnings guidance"},
        frequency=DataFrequency.IRREGULAR,
        source_id="test-source",
    )
    source = _SingleRecordSource(record)

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

    for index, delay in enumerate(trainer_delays, start=1):
        model_id = f"news-model-{index}"
        model_registry.register(
            ModelEntryConfig(
                model_id=model_id,
                model_type="ml",
                signal_type=SignalType.SENTIMENT,
                trainer_class="tests.mocks.mock_ml_trainer.MockMLTrainer",
                input_data_types=[DataType.NEWS_TEXT],
                input_frequency=DataFrequency.IRREGULAR,
            )
        )
        entry = model_registry.get(model_id)
        assert entry is not None
        entry.trainer = _NewsTrainer(value=0.3 * index, delay_seconds=delay)

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
        cache_config={"max_age_seconds": 600.0, "max_signals_per_model": 20},
    )
    pipeline.configure(
        {
            "max_workers": 4,
            "default_timeout_seconds": 1.0,
            "prune_every_n_results": 100,
        }
    )

    return pipeline, router, model_registry, strategy_registry


def test_pipeline_routes_data_and_caches_latest_signals(tmp_path) -> None:
    pipeline, router, model_registry, strategy_registry = _build_pipeline(tmp_path)
    del model_registry, strategy_registry

    received: list[tuple[str, ModelSignal]] = []
    pipeline.add_signal_listener(
        lambda model_id, signal: received.append((model_id, signal))
    )

    pipeline.start()
    router.start()

    assert _wait_until(lambda: len(pipeline.get_all_latest_signals()) >= 2)

    latest = pipeline.get_all_latest_signals()
    assert len(latest) == 2
    assert any(model_id.startswith("news-model-") for model_id in latest)

    metrics = pipeline.get_metrics()
    assert metrics.total_inferences == 2
    assert metrics.successful_inferences == 2
    assert metrics.failed_inferences == 0

    assert len(received) == 2

    router.stop()
    pipeline.stop()


def test_pipeline_runs_multiple_models_concurrently(tmp_path) -> None:
    pipeline, router, model_registry, strategy_registry = _build_pipeline(
        tmp_path,
        trainer_delays=(0.3, 0.3),
    )
    del model_registry, strategy_registry

    pipeline.start()
    start = time.perf_counter()
    router.start()

    assert _wait_until(
        lambda: len(pipeline.get_all_latest_signals()) >= 2,
        timeout_seconds=4.0,
    )
    elapsed = time.perf_counter() - start

    # Two model delays (0.3 + 0.3) should complete under serial sum when concurrent.
    assert elapsed < 0.55

    router.stop()
    pipeline.stop()


def test_pipeline_metrics_capture_failures(tmp_path) -> None:
    pipeline, router, model_registry, strategy_registry = _build_pipeline(tmp_path)
    del strategy_registry

    bad_id = "news-model-2"
    bad_entry = model_registry.get(bad_id)
    assert bad_entry is not None

    class _BadTrainer:
        def predict(self, value: object) -> float:
            del value
            raise RuntimeError("inference failure")

    bad_entry.trainer = _BadTrainer()

    pipeline.start()
    router.start()

    assert _wait_until(lambda: pipeline.get_metrics().total_inferences >= 2)

    metrics = pipeline.get_metrics()
    assert metrics.total_inferences == 2
    assert metrics.successful_inferences == 1
    assert metrics.failed_inferences == 1
    assert metrics.errors_by_model[bad_id] == 1

    router.stop()
    pipeline.stop()
