"""Cross-component integration tests for the supporting-model ecosystem."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from algotrading.src.data_pipeline import (
    DataBatch,
    DataFrequency,
    DataRecord,
    DataType,
    SourceMetadata,
)
from algotrading.src.data_pipeline.routing import DataRouter, RouteConfig, RouterConfig
from algotrading.src.data_pipeline.sources.base import DataSource
from algotrading.src.envs.signal_integration import SignalConfig, SignalIntegration
from algotrading.src.envs.trading_env import TradingEnv
from algotrading.src.models.inference import (
    AlignmentConfig,
    InferencePipeline,
    SignalCache,
    TimestampAlignmentService,
)
from algotrading.src.models.registry import (
    CustomStrategyRegistry,
    ModelEntryConfig,
    ModelState,
    SupportingModelRegistry,
)
from algotrading.src.models.signals import SignalType
from algotrading.src.models.tracking import (
    EvaluationMetrics,
    GenerationTracker,
    JsonFileStorage,
    TrainingMetrics,
)

from tests.fixtures.mock_supporting_model import MockSupportingStrategy


class _SingleRecordSource(DataSource):
    """Source emitting exactly one deterministic record for integration tests."""

    def __init__(self, record: DataRecord) -> None:
        self._record = record
        self._connected = False

    @property
    def source_id(self) -> str:
        return "phase4-supporting-model-source"

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
        records = []
        if (
            symbol == self._record.symbol
            and data_type == self._record.data_type
            and start <= self._record.timestamp <= end
        ):
            records = [self._record]

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


@dataclass
class _CustomLogic:
    """Minimal reward logic used by integration environment state-builder double."""

    CUSTOM_VARIABLES = {"current_position": (0.0, 1.0, np.float64)}

    def calculate_reward(self, state: dict[str, Any]) -> float:
        """Return deterministic reward for integration assertions."""

        _ = state
        return 1.0

    def reset_env_globals(self) -> None:
        """No-op reset implementation used by BaseTradingEnv."""


class _StateBuilder:
    """StateBuilder test double for signal-enhanced environment integration."""

    def __init__(self, timestamp: datetime) -> None:
        self.custom_logic = _CustomLogic()
        self.pipeline = {
            "pipeline": {
                "state_data_config": {
                    "columns": {
                        "date": [True, False],
                        "open": [False, True],
                        "high": [False, True],
                        "low": [False, True],
                        "close": [False, True],
                        "volume": [False, True],
                        "count": [False, True],
                    }
                },
                "trading_config": {"stop_take": {"enabled": False}},
            }
        }
        self.state_counters = {"step": 0, "window": 0, "episode": 1}
        self.terminated = False
        self.timed_out = False
        self.state = self._build_state()
        self.state_df = pd.DataFrame({"date": [timestamp]})

    def _build_state(self) -> dict[str, np.ndarray]:
        return {
            "open": np.array([100.0, 101.0, 102.0]),
            "high": np.array([101.0, 102.0, 103.0]),
            "low": np.array([99.0, 100.0, 101.0]),
            "close": np.array([100.5, 101.5, 102.5]),
            "volume": np.array([1000.0, 980.0, 1010.0]),
            "count": np.array([10.0, 11.0, 9.0]),
            "current_position": np.array([0.0, 1.0, 0.0]),
        }

    def state_step(self, action: int) -> None:
        """Advance environment timestamp by one bar for each step."""

        del action
        self.state_counters["step"] += 1
        next_ts = self.state_df["date"].iloc[-1] + pd.Timedelta(seconds=5)
        self.state_df = pd.DataFrame({"date": [next_ts]})

    def update_episode_counter(self) -> None:
        """Increment episode counter for BaseTradingEnv compatibility."""

        self.state_counters["episode"] += 1

    def initialise_state(self) -> None:
        """Reset state payload arrays to initial deterministic values."""

        self.state = self._build_state()


def _wait_until(predicate, timeout_seconds: float = 5.0) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_end_to_end_supporting_model_flow_from_registry_to_environment(
    tmp_path,
) -> None:
    """Model and strategy signals should flow through pipeline, alignment, and env."""

    timestamp = datetime.now(tz=UTC)
    record = DataRecord(
        timestamp=timestamp,
        data_type=DataType.MARKET_BAR,
        symbol="AAPL",
        payload={
            "open": 100.0,
            "close": 101.2,
            "headline": "Company beats expectations and raises guidance",
        },
        frequency=DataFrequency.SECOND_5,
        source_id="phase4-test-source",
    )
    source = _SingleRecordSource(record)
    router = DataRouter(
        RouterConfig(
            routes=[
                RouteConfig(
                    source=source,
                    data_types=[DataType.MARKET_BAR],
                    symbols=["AAPL"],
                    targets=["callback"],
                )
            ]
        )
    )

    model_registry = SupportingModelRegistry()
    model_id = "mock-supporting-model"
    model_registry.register(
        ModelEntryConfig(
            model_id=model_id,
            model_type="ml",
            signal_type=SignalType.SENTIMENT,
            trainer_class="tests.fixtures.mock_supporting_model.MockSupportingModel",
            input_data_types=[DataType.MARKET_BAR],
            input_frequency=DataFrequency.SECOND_5,
            description="Deterministic mock supporting model",
        )
    )
    model_registry.load_model(model_id)
    model_registry.set_state(model_id, ModelState.READY)

    strategy_registry = CustomStrategyRegistry(strategy_dirs=[], auto_scan=False)
    strategy_id = strategy_registry.register_strategy(
        MockSupportingStrategy,
        filepath=Path(__file__).resolve(),
    )

    pipeline = InferencePipeline(
        model_registry=model_registry,
        strategy_registry=strategy_registry,
        data_router=router,
    )
    pipeline.configure(
        {
            "max_workers": 4,
            "default_timeout_seconds": 1.0,
            "prune_every_n_results": 100,
        }
    )

    pipeline.start()
    router.start()

    assert _wait_until(lambda: len(pipeline.get_all_latest_signals()) == 2)

    latest = pipeline.get_all_latest_signals()
    assert set(latest) == {model_id, strategy_id}
    assert latest[model_id].signal_type == SignalType.SENTIMENT
    assert latest[strategy_id].signal_type == SignalType.POSITION

    alignment_cache = SignalCache(max_age_seconds=300.0, max_signals_per_model=20)
    for cache_model_id, signal in latest.items():
        alignment_cache.put(cache_model_id, signal)

    alignment = TimestampAlignmentService(signal_cache=alignment_cache)
    alignment.configure_model(
        model_id,
        AlignmentConfig(model_id=model_id, max_staleness_seconds=120),
    )
    alignment.configure_model(
        strategy_id,
        AlignmentConfig(model_id=strategy_id, max_staleness_seconds=120),
    )

    aligned = alignment.align(timestamp + timedelta(seconds=1), [model_id, strategy_id])
    assert aligned.signals[model_id].signal is not None
    assert aligned.signals[strategy_id].signal is not None

    integration = SignalIntegration(
        alignment_service=alignment,
        signal_configs=[
            SignalConfig(
                model_id=model_id,
                signal_key="sentiment",
                default_value=0.0,
                include_confidence=True,
                include_staleness=True,
            ),
            SignalConfig(
                model_id=strategy_id,
                signal_key="position",
                default_value=0.0,
                include_confidence=True,
                include_staleness=True,
            ),
        ],
    )

    env = TradingEnv(_StateBuilder(timestamp), signal_integration=integration)
    obs, info = env.reset()
    assert info == {}
    assert "market" in obs
    assert "signals" in obs
    assert obs["signals"].shape == (6,)

    step_obs, reward, terminated, truncated, step_info = env.step(1)
    assert "signals" in step_obs
    assert step_obs["signals"].shape == (6,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "signals" in step_info
    assert set(step_info["signals"]) == {"sentiment", "position"}

    router.stop()
    pipeline.stop()


def test_generation_tracking_lifecycle_for_supporting_model(tmp_path) -> None:
    """Supporting model generations should track lifecycle and comparisons."""

    tracker = GenerationTracker(JsonFileStorage(str(tmp_path / "generations")))

    gen_1 = tracker.start_generation(
        model_id="mock-supporting-model",
        hyperparameters={"learning_rate": 0.001},
    )
    tracker.complete_generation(
        gen_1.generation_id,
        TrainingMetrics(
            final_reward=0.7,
            mean_reward=0.5,
            std_reward=0.2,
            episodes_completed=6,
            timesteps_trained=1200,
            training_time_seconds=3.5,
        ),
        model_path="model_gen_1.pkl",
    )
    tracker.add_evaluation(
        gen_1.generation_id,
        EvaluationMetrics(
            sharpe_ratio=0.9,
            max_drawdown=0.2,
            total_return=0.13,
            win_rate=0.55,
            profit_factor=1.15,
            num_trades=18,
        ),
    )

    gen_2 = tracker.start_generation(
        model_id="mock-supporting-model",
        hyperparameters={"learning_rate": 0.0005},
        parent_generation_id=gen_1.generation_id,
    )
    tracker.complete_generation(
        gen_2.generation_id,
        TrainingMetrics(
            final_reward=1.1,
            mean_reward=0.8,
            std_reward=0.15,
            episodes_completed=8,
            timesteps_trained=1400,
            training_time_seconds=4.1,
        ),
        model_path="model_gen_2.pkl",
    )
    tracker.add_evaluation(
        gen_2.generation_id,
        EvaluationMetrics(
            sharpe_ratio=1.3,
            max_drawdown=0.14,
            total_return=0.22,
            win_rate=0.62,
            profit_factor=1.31,
            num_trades=24,
        ),
    )

    latest = tracker.get_latest_generation("mock-supporting-model")
    best = tracker.get_best_generation("mock-supporting-model", "sharpe_ratio")
    comparison = tracker.compare_generations(gen_1.generation_id, gen_2.generation_id)

    assert latest is not None
    assert latest.generation_id == gen_2.generation_id
    assert best is not None
    assert best.generation_id == gen_2.generation_id
    assert comparison.metric_comparisons["sharpe_ratio"][2] == 0.4
