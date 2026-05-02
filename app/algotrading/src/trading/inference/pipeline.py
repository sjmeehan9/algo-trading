"""Real-time market-data to trading-decision inference pipeline."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from concurrent.futures import Future, wait
from datetime import UTC, datetime
from math import isfinite
from typing import cast

from algotrading.src.broker import BarData
from algotrading.src.data_pipeline import DataFrequency, DataRecord, DataType
from algotrading.src.models.inference import InferencePipeline, InferenceResult
from algotrading.src.models.registry import SupportingModelRegistry
from algotrading.src.models.registry.model_entry import ModelState
from algotrading.src.models.signals import ModelSignal
from algotrading.src.trading.deployment import validate_core_model_instance
from algotrading.src.trading.inference.aggregator import AlignedSignal, SignalAggregator
from algotrading.src.trading.inference.decision import TradingDecision
from algotrading.src.trading.inference.latency import LatencyTracker
from algotrading.src.trainers import RLTrainer

logger = logging.getLogger(__name__)


class RealTimeInferencePipeline:
    """Orchestrate live market data, supporting signals, and core RL inference."""

    def __init__(
        self,
        core_model: RLTrainer,
        supporting_registry: SupportingModelRegistry,
        signal_aggregator: SignalAggregator,
        latency_tracker: LatencyTracker,
        config: dict[str, object] | None = None,
        supporting_inference_pipeline: InferencePipeline | None = None,
    ) -> None:
        """Initialize the real-time inference pipeline.

        Args:
            core_model: Trained core RL model used to produce trading actions.
            supporting_registry: Registry of supporting models that may emit signals.
            signal_aggregator: Timestamp alignment and fallback aggregation service.
            latency_tracker: Stage timing collector.
            config: Optional runtime settings.
            supporting_inference_pipeline: Optional Phase 4 pipeline used to run
                supporting model inference when market bars arrive.
        """

        self.core_model = core_model
        self.supporting_registry = supporting_registry
        self.signal_aggregator = signal_aggregator
        self.latency_tracker = latency_tracker
        self.config = dict(config or {})
        self.supporting_inference_pipeline = supporting_inference_pipeline

        validate_core_model_instance(
            self.core_model,
            configured_model_type=self._configured_deployment_model_type(),
        )

        self.stale_threshold = float(self.config.get("stale_signal_threshold", 60.0))
        if self.stale_threshold <= 0 or not isfinite(self.stale_threshold):
            raise ValueError("stale_signal_threshold must be finite and greater than 0")

        self.latency_target_ms = float(self.config.get("latency_target_ms", 100.0))
        if self.latency_target_ms <= 0 or not isfinite(self.latency_target_ms):
            raise ValueError("latency_target_ms must be finite and greater than 0")

        self.supporting_signal_wait_ms = float(
            self.config.get("supporting_signal_wait_ms", 0.0)
        )
        if self.supporting_signal_wait_ms < 0 or not isfinite(
            self.supporting_signal_wait_ms
        ):
            raise ValueError(
                "supporting_signal_wait_ms must be finite and non-negative"
            )

        self._running = False
        self._last_decision: TradingDecision | None = None
        self._decision_callbacks: list[Callable[[TradingDecision], None]] = []

        if hasattr(self.core_model, "ensure_trained"):
            self.core_model.ensure_trained()

    def _configured_deployment_model_type(self) -> object | None:
        """Return an explicit deployment model category from runtime config."""

        for key in ("deployment_model_type", "model_category", "root_model_type"):
            value = self.config.get(key)
            if value is not None:
                return value

        model_type = self.config.get("model_type")
        if isinstance(model_type, str) and model_type in {
            "core_rl",
            "supporting_ml",
            "supporting_rl",
            "strategy",
        }:
            return model_type

        return None

    async def start(
        self,
        decision_callback: Callable[[TradingDecision], None] | None = None,
    ) -> None:
        """Start the inference pipeline and optional supporting pipeline."""

        if decision_callback is not None:
            self.add_decision_callback(decision_callback)

        if self.supporting_inference_pipeline is not None:
            self.supporting_inference_pipeline.start()

        self._running = True
        logger.info("Real-time inference pipeline started")

    async def stop(self) -> None:
        """Stop the inference pipeline and optional supporting pipeline."""

        self._running = False
        if self.supporting_inference_pipeline is not None:
            self.supporting_inference_pipeline.stop()
        logger.info("Real-time inference pipeline stopped")

    def start_sync(
        self,
        decision_callback: Callable[[TradingDecision], None] | None = None,
    ) -> None:
        """Synchronously start the pipeline for legacy callback contexts."""

        self._run_coro_sync(self.start(decision_callback=decision_callback))

    def stop_sync(self) -> None:
        """Synchronously stop the pipeline for legacy callback contexts."""

        self._run_coro_sync(self.stop())

    def is_running(self) -> bool:
        """Return whether the pipeline is currently accepting market bars."""

        return self._running

    def add_decision_callback(
        self, callback: Callable[[TradingDecision], None]
    ) -> None:
        """Register a callback invoked after each generated decision."""

        self._decision_callbacks.append(callback)

    def get_last_decision(self) -> TradingDecision | None:
        """Return the most recent trading decision."""

        return self._last_decision

    async def process_market_data(self, bar_data: BarData) -> TradingDecision:
        """Process one broker bar through supporting and core inference.

        Args:
            bar_data: Broker-normalized OHLCV bar.

        Returns:
            Trading decision generated by the core RL model.
        """

        if not self._running:
            raise RuntimeError("Real-time inference pipeline is not running")

        total_measurement = None
        decision: TradingDecision | None = None

        with self.latency_tracker.track("total_pipeline") as total_measurement:
            timestamp = self._ensure_aware_timestamp(bar_data.timestamp)

            with self.latency_tracker.track("market_state"):
                market_state = self._build_market_state(bar_data)
                market_record = self._market_state_to_record(market_state, bar_data)

            with self.latency_tracker.track("supporting_trigger"):
                futures = self._trigger_supporting_inference(market_record)
                self._wait_for_supporting_futures(futures)

            with self.latency_tracker.track("supporting_signals"):
                signals = await self._get_supporting_signals(timestamp)

            with self.latency_tracker.track("signal_aggregation"):
                aligned_signals = self.signal_aggregator.aggregate(
                    signals=signals,
                    target_timestamp=timestamp,
                    stale_threshold=self.stale_threshold,
                    expected_signal_ids=self._get_expected_signal_ids(),
                )

            with self.latency_tracker.track("observation_build"):
                observation = self._build_observation(market_state, aligned_signals)

            with self.latency_tracker.track("core_inference"):
                action, info = self.core_model.predict(
                    observation,
                    deterministic=bool(self.config.get("deterministic", True)),
                )

            decision = TradingDecision(
                timestamp=timestamp,
                action=self._action_to_trading_action(action),
                confidence=self._extract_confidence(info),
                market_price=bar_data.close,
                signals_used=[
                    signal_id
                    for signal_id, aligned in aligned_signals.items()
                    if not aligned.is_stale and not aligned.used_default
                ],
                latency_ms=0.0,
                metadata={
                    "stale_signals": [
                        signal_id
                        for signal_id, aligned in aligned_signals.items()
                        if aligned.is_stale
                    ],
                    "defaulted_signals": [
                        signal_id
                        for signal_id, aligned in aligned_signals.items()
                        if aligned.used_default
                    ],
                    "signal_count": len(aligned_signals),
                },
            )

        if decision is None or total_measurement is None:
            raise RuntimeError("Pipeline did not produce a trading decision")

        decision.latency_ms = total_measurement.elapsed_ms
        self._last_decision = decision
        self._notify_decision_callbacks(decision)

        if decision.latency_ms > self.latency_target_ms:
            logger.warning(
                "Real-time inference latency %.2fms exceeded target %.2fms",
                decision.latency_ms,
                self.latency_target_ms,
            )

        return decision

    def process_market_data_sync(self, bar_data: BarData) -> TradingDecision:
        """Synchronously process one broker bar from legacy trading callbacks."""

        return cast(
            TradingDecision,
            self._run_coro_sync(self.process_market_data(bar_data)),
        )

    def get_latency_stats(self) -> dict[str, object]:
        """Return latency statistics for all tracked pipeline stages."""

        return self.latency_tracker.get_stats()

    def _build_market_state(self, bar_data: BarData) -> dict[str, object]:
        """Build model-ready market state from broker bar data."""

        timestamp = self._ensure_aware_timestamp(bar_data.timestamp)
        return {
            "timestamp": timestamp,
            "open": float(bar_data.open),
            "high": float(bar_data.high),
            "low": float(bar_data.low),
            "close": float(bar_data.close),
            "volume": int(bar_data.volume),
            "vwap": float(bar_data.vwap) if bar_data.vwap is not None else 0.0,
            "trade_count": int(bar_data.trade_count or 0),
        }

    def _market_state_to_record(
        self,
        market_state: dict[str, object],
        bar_data: BarData,
    ) -> DataRecord:
        """Convert market state to a data-router record for supporting models."""

        symbol = str(self.config.get("symbol", "UNKNOWN"))
        return DataRecord(
            timestamp=cast(datetime, market_state["timestamp"]),
            data_type=DataType.MARKET_BAR,
            symbol=symbol,
            payload={
                "open": market_state["open"],
                "high": market_state["high"],
                "low": market_state["low"],
                "close": market_state["close"],
                "volume": market_state["volume"],
                "vwap": market_state["vwap"],
                "trade_count": market_state["trade_count"],
                "bar": bar_data,
            },
            source_id=str(self.config.get("market_source_id", "broker")),
            frequency=self._resolve_market_frequency(),
        )

    def _trigger_supporting_inference(
        self,
        market_record: DataRecord,
    ) -> list[Future[InferenceResult]]:
        """Submit market data to the supporting inference pipeline if configured."""

        if self.supporting_inference_pipeline is None:
            return []
        return self.supporting_inference_pipeline.submit_record(market_record)

    def _wait_for_supporting_futures(
        self,
        futures: list[Future[InferenceResult]],
    ) -> None:
        """Optionally wait briefly for fresh supporting model outputs."""

        if not futures or self.supporting_signal_wait_ms <= 0:
            return

        deadline = time.perf_counter() + (self.supporting_signal_wait_ms / 1000.0)
        remaining = max(0.0, deadline - time.perf_counter())
        if remaining > 0:
            wait(futures, timeout=remaining)

    async def _get_supporting_signals(
        self,
        timestamp: datetime,
    ) -> dict[str, ModelSignal]:
        """Return latest supporting signals available at or before timestamp."""

        if self.supporting_inference_pipeline is None:
            return {}

        signals: dict[str, ModelSignal] = {}
        for model_id in self._get_expected_signal_ids():
            signal = self.supporting_inference_pipeline.get_signal_before(
                model_id,
                timestamp,
            )
            if signal is not None:
                signals[model_id] = signal

        for (
            model_id,
            signal,
        ) in self.supporting_inference_pipeline.get_all_latest_signals().items():
            if signal.timestamp <= timestamp:
                signals.setdefault(model_id, signal)

        return signals

    def _build_observation(
        self,
        market_state: dict[str, object],
        signals: dict[str, AlignedSignal],
    ) -> dict[str, object]:
        """Construct core RL observation with market state and signal features."""

        return {
            "market": market_state,
            "signals": {
                signal_id: aligned_signal.value_for_observation
                for signal_id, aligned_signal in signals.items()
            },
            "signal_confidence": {
                signal_id: aligned_signal.confidence
                for signal_id, aligned_signal in signals.items()
            },
            "signal_staleness": {
                signal_id: aligned_signal.is_stale
                for signal_id, aligned_signal in signals.items()
            },
        }

    def _get_expected_signal_ids(self) -> list[str]:
        """Resolve supporting signal IDs that should be present in observations."""

        configured = self.config.get("supporting_model_ids")
        if isinstance(configured, list):
            return [str(item) for item in configured if str(item).strip()]

        ready_ids = [
            entry.config.model_id for entry in self.supporting_registry.find_ready()
        ]
        if ready_ids:
            return sorted(ready_ids)

        default_ids = self.signal_aggregator.get_registered_signal_ids()
        if default_ids:
            return sorted(default_ids)

        return [
            entry.config.model_id
            for entry in self.supporting_registry.get_all()
            if entry.state in {ModelState.LOADED, ModelState.READY}
        ]

    def _action_to_trading_action(self, action: object) -> str:
        """Convert core model action output to a trading action string."""

        action_value = int(action)
        action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        return action_map.get(action_value, "HOLD")

    def _extract_confidence(self, info: dict[str, object]) -> float:
        """Extract a bounded confidence score from model prediction metadata."""

        raw_confidence = info.get("confidence", 1.0)
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            logger.debug("Invalid confidence value from core model: %r", raw_confidence)
            return 1.0
        if not isfinite(confidence):
            return 1.0
        return max(0.0, min(1.0, confidence))

    def _resolve_market_frequency(self) -> DataFrequency:
        """Resolve market data frequency from config for data-router records."""

        raw_frequency = self.config.get("market_data_frequency")
        if isinstance(raw_frequency, DataFrequency):
            return raw_frequency
        if isinstance(raw_frequency, str) and raw_frequency.strip():
            return DataFrequency(raw_frequency.strip())

        bar_size_seconds = int(self.config.get("bar_size_seconds", 5))
        mapping = {
            1: DataFrequency.SECOND_1,
            5: DataFrequency.SECOND_5,
            10: DataFrequency.SECOND_10,
            30: DataFrequency.SECOND_30,
            60: DataFrequency.MINUTE_1,
            300: DataFrequency.MINUTE_5,
            900: DataFrequency.MINUTE_15,
            3600: DataFrequency.HOUR_1,
        }
        return mapping.get(bar_size_seconds, DataFrequency.IRREGULAR)

    def _notify_decision_callbacks(self, decision: TradingDecision) -> None:
        """Notify registered decision callbacks without interrupting trading."""

        for callback in list(self._decision_callbacks):
            try:
                callback(decision)
            except Exception:
                logger.exception("Trading decision callback failed")

    def _ensure_aware_timestamp(self, timestamp: datetime) -> datetime:
        """Ensure broker timestamps are timezone-aware."""

        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            return timestamp.replace(tzinfo=UTC)
        return timestamp

    def _run_coro_sync(self, coro: Coroutine[object, object, object]) -> object:
        """Run an async coroutine from synchronous trading code."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        coro.close()
        raise RuntimeError(
            "Synchronous real-time inference cannot run inside an active event loop"
        )
