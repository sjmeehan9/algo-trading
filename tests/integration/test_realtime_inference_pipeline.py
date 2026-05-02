"""Integration tests for the real-time trading inference pipeline."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from algotrading.src.broker import BarData
from algotrading.src.data_pipeline import DataFrequency, DataType
from algotrading.src.data_pipeline.routing import DataRouter
from algotrading.src.models.inference import (
    InferencePipeline,
    SignalCache,
    TimestampAlignmentService,
)
from algotrading.src.models.registry import (
    CustomStrategyRegistry,
    ModelEntryConfig,
    SupportingModelRegistry,
)
from algotrading.src.models.signals import ModelSignal, SignalMetadata, SignalType
from algotrading.src.trading.inference import (
    LatencyTracker,
    RealTimeInferencePipeline,
    SignalAggregator,
)
from algotrading.src.trainers import (
    EvaluationResult,
    RLTrainer,
    TrainingConfig,
    TrainingResult,
)
from gymnasium import Env


class _CoreTrainer(RLTrainer):
    """Minimal trained RL trainer used for real-time inference tests."""

    def __init__(self, action: int = 1, confidence: float = 0.9) -> None:
        self.action = action
        self.confidence = confidence
        self.last_observation: dict[str, object] | None = None

    def create_model(self, env: Env, config: TrainingConfig) -> None:
        """No-op model creation for test trainer."""

        del env, config

    def train(
        self,
        config: TrainingConfig,
        callback: Callable[[dict[str, object]], None] | None = None,
    ) -> TrainingResult:
        """Return a deterministic training result."""

        del config, callback
        return TrainingResult(0, 0, 0.0, 0.0)

    def evaluate(self, env: Env, n_episodes: int = 10) -> EvaluationResult:
        """Return deterministic evaluation metrics."""

        del env, n_episodes
        return EvaluationResult(0, 0.0, 0.0, 0.0)

    def save(self, filepath: str) -> None:
        """No-op save for test trainer."""

        del filepath

    def load(self, filepath: str, env: Env | None = None) -> None:
        """No-op load for test trainer."""

        del filepath, env

    def predict(
        self,
        observation: dict[str, object],
        deterministic: bool = True,
    ) -> tuple[int, dict[str, object]]:
        """Store observation and return configured action/confidence."""

        del deterministic
        self.last_observation = observation
        return self.action, {"confidence": self.confidence}

    @property
    def model_type(self) -> str:
        """Return model type identifier."""

        return "mock_core_rl"

    @property
    def is_trained(self) -> bool:
        """Return that the test model is ready for inference."""

        return True


class _MarketSignalTrainer:
    """Supporting model trainer that emits a scalar from market payloads."""

    def __init__(self, value: float) -> None:
        self.value = value
        self.payloads: list[object] = []

    def predict(self, value: object) -> float:
        """Record payload and return configured signal value."""

        self.payloads.append(value)
        return self.value


def _bar(timestamp: datetime | None = None) -> BarData:
    """Build a valid broker bar for integration tests."""

    return BarData(
        timestamp=timestamp or datetime.now(tz=UTC),
        open=100.0,
        high=102.0,
        low=99.5,
        close=101.0,
        volume=10_000,
        vwap=100.75,
        trade_count=42,
    )


def _signal(model_id: str, timestamp: datetime, value: float) -> ModelSignal:
    """Build a cached supporting signal."""

    return ModelSignal(
        timestamp=timestamp,
        signal_type=SignalType.SENTIMENT,
        value=value,
        confidence=0.6,
        symbol="AAPL",
        metadata=SignalMetadata(model_id=model_id, model_type="ml"),
    )


def _empty_registry() -> SupportingModelRegistry:
    """Create an empty supporting model registry."""

    return SupportingModelRegistry()


async def _started_pipeline(
    pipeline: RealTimeInferencePipeline,
) -> RealTimeInferencePipeline:
    """Start a real-time pipeline and return it."""

    await pipeline.start()
    return pipeline


@pytest.mark.asyncio
async def test_full_pipeline_triggers_supporting_model_and_builds_decision(
    tmp_path,
) -> None:
    """Broker market bars should trigger supporting inference and RL decisions."""

    registry = SupportingModelRegistry()
    registry.register(
        ModelEntryConfig(
            model_id="market-signal",
            model_type="ml",
            signal_type=SignalType.SENTIMENT,
            trainer_class="tests.mocks.mock_ml_trainer.MockMLTrainer",
            input_data_types=[DataType.MARKET_BAR],
            input_frequency=DataFrequency.SECOND_5,
        )
    )
    trainer = _MarketSignalTrainer(value=0.7)
    entry = registry.get("market-signal")
    assert entry is not None
    entry.trainer = trainer

    supporting_pipeline = InferencePipeline(
        model_registry=registry,
        strategy_registry=CustomStrategyRegistry(
            strategy_dirs=[str(tmp_path / "strategies")],
            auto_scan=True,
        ),
        data_router=DataRouter(),
    )
    supporting_pipeline.configure({"max_workers": 2, "default_timeout_seconds": 1.0})

    aggregator = SignalAggregator(TimestampAlignmentService(SignalCache()))
    aggregator.register_signal_default("market-signal", 0.0)
    core_model = _CoreTrainer(action=1, confidence=0.92)
    pipeline = RealTimeInferencePipeline(
        core_model=core_model,
        supporting_registry=registry,
        signal_aggregator=aggregator,
        latency_tracker=LatencyTracker(warning_threshold_ms=1000.0),
        config={
            "symbol": "AAPL",
            "supporting_model_ids": ["market-signal"],
            "supporting_signal_wait_ms": 50.0,
            "latency_target_ms": 100.0,
        },
        supporting_inference_pipeline=supporting_pipeline,
    )

    await _started_pipeline(pipeline)
    try:
        decision = await pipeline.process_market_data(_bar())
    finally:
        await pipeline.stop()

    assert decision.action == "BUY"
    assert decision.confidence == 0.92
    assert decision.signals_used == ["market-signal"]
    assert decision.latency_ms < 100.0
    assert trainer.payloads
    assert core_model.last_observation is not None
    assert core_model.last_observation["signals"] == {"market-signal": 0.7}
    assert core_model.last_observation["signal_staleness"] == {"market-signal": False}


@pytest.mark.asyncio
async def test_pipeline_handles_missing_supporting_signal_with_default() -> None:
    """Missing expected signals should not block decision generation."""

    registry = _empty_registry()
    aggregator = SignalAggregator(TimestampAlignmentService(SignalCache()))
    aggregator.register_signal_default("missing-signal", 0.0)
    core_model = _CoreTrainer(action=0, confidence=1.0)
    pipeline = RealTimeInferencePipeline(
        core_model=core_model,
        supporting_registry=registry,
        signal_aggregator=aggregator,
        latency_tracker=LatencyTracker(warning_threshold_ms=1000.0),
        config={"supporting_model_ids": ["missing-signal"]},
    )

    await _started_pipeline(pipeline)
    try:
        decision = await pipeline.process_market_data(_bar())
    finally:
        await pipeline.stop()

    assert decision.action == "HOLD"
    assert decision.signals_used == []
    assert "missing-signal" in decision.metadata["defaulted_signals"]
    assert core_model.last_observation is not None
    assert core_model.last_observation["signals"] == {"missing-signal": 0.0}
    assert core_model.last_observation["signal_staleness"] == {"missing-signal": True}


@pytest.mark.asyncio
async def test_pipeline_uses_default_for_stale_aligned_signal() -> None:
    """Stale cache signals should be marked and replaced in observations."""

    now = datetime.now(tz=UTC)
    cache = SignalCache(max_age_seconds=600.0)
    cache.put("old-signal", _signal("old-signal", now - timedelta(seconds=120), 0.9))

    aggregator = SignalAggregator(TimestampAlignmentService(cache))
    aggregator.register_signal_default("old-signal", -0.2)
    core_model = _CoreTrainer(action=2, confidence=0.8)
    pipeline = RealTimeInferencePipeline(
        core_model=core_model,
        supporting_registry=_empty_registry(),
        signal_aggregator=aggregator,
        latency_tracker=LatencyTracker(warning_threshold_ms=1000.0),
        config={
            "supporting_model_ids": ["old-signal"],
            "stale_signal_threshold": 30.0,
        },
    )

    await _started_pipeline(pipeline)
    try:
        decision = await pipeline.process_market_data(_bar(timestamp=now))
    finally:
        await pipeline.stop()

    assert decision.action == "SELL"
    assert decision.signals_used == []
    assert "old-signal" in decision.metadata["stale_signals"]
    assert core_model.last_observation is not None
    assert core_model.last_observation["signals"] == {"old-signal": -0.2}
    assert core_model.last_observation["signal_staleness"] == {"old-signal": True}


def test_trading_integration_uses_inference_decision(
    mock_config: dict,
    mock_pipeline_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trading should consume injected real-time decisions through existing flow."""

    from algotrading.src.trading.inference import TradingDecision
    from algotrading.src.trading.trading import Trading

    from tests.unit.test_trading_adapter_refactor import _MockBrokerAdapter, _TimerStub

    class _PipelineStub:
        """Synchronous real-time pipeline stub for Trading integration."""

        def __init__(self) -> None:
            self.started = False
            self.stopped = False
            self.last_bar: BarData | None = None

        def is_running(self) -> bool:
            """Return current running state."""

            return self.started and not self.stopped

        def start_sync(self) -> None:
            """Mark the stub started."""

            self.started = True

        def stop_sync(self) -> None:
            """Mark the stub stopped."""

            self.stopped = True

        def process_market_data_sync(self, bar_data: BarData) -> TradingDecision:
            """Return a deterministic buy decision."""

            self.last_bar = bar_data
            return TradingDecision(
                timestamp=bar_data.timestamp,
                action="BUY",
                confidence=0.9,
                market_price=bar_data.close,
                signals_used=[],
                latency_ms=1.0,
            )

    monkeypatch.setattr("algotrading.src.trading.trading.Timer", _TimerStub)

    config = dict(mock_config)
    config["stream_data"] = "real"
    pipeline_config = dict(mock_pipeline_config)
    pipeline_config["pipeline"] = dict(mock_pipeline_config["pipeline"])
    pipeline_config["pipeline"]["trading_config"] = dict(
        mock_pipeline_config["pipeline"]["trading_config"]
    )
    pipeline_config["pipeline"]["trading_config"]["stop_take"] = dict(
        mock_pipeline_config["pipeline"]["trading_config"]["stop_take"]
    )
    pipeline_config["pipeline"]["trading_config"]["stop_take"]["enabled"] = False

    stub = _PipelineStub()
    trading = Trading(
        config,
        pipeline_config,
        adapter=_MockBrokerAdapter(),
        inference_pipeline=stub,  # type: ignore[arg-type]
    )
    trading.order.checkAction = lambda action, active_pos: action == "BUY"
    trading.order.calcOrderSpec = lambda *args, **kwargs: ["BUY", 1, "open"]

    state = {"current_position": [0], "trade_change": [0.0]}
    state_df = pd.DataFrame(
        [
            {
                "date": datetime.now(tz=UTC),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000,
                "wap": 100.2,
                "count": 10,
            }
        ]
    )

    trading.tradingAlgorithm(state, state_df)

    assert stub.started is True
    assert stub.last_bar is not None
    assert trading.payload.action_str == "BUY"
    assert trading.payload.active_pos == "BUY_PEND"
    assert trading.payload.last_decision.action == "BUY"

    trading.stop()
    assert stub.stopped is True
