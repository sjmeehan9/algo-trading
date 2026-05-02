"""Asynchronous inference pipeline for supporting models and strategies."""

from __future__ import annotations

import logging
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock
from typing import Callable
from uuid import uuid4

from algotrading.src.data_pipeline import DataRecord, DataType, NewsRecord
from algotrading.src.data_pipeline.routing import DataRouter
from algotrading.src.data_pipeline.sources import NewsDataSource
from algotrading.src.models.inference.cache import SignalCache
from algotrading.src.models.inference.executor import (
    InferenceExecutor,
    InferenceResult,
    InferenceTask,
)
from algotrading.src.models.registry import (
    CustomStrategyRegistry,
    ModelEntry,
    SupportingModelRegistry,
)
from algotrading.src.models.registry.strategy_registry import StrategyEntry
from algotrading.src.models.signals import ModelSignal

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class InferenceMetrics:
    """Runtime statistics for inference throughput and reliability."""

    total_inferences: int = 0
    successful_inferences: int = 0
    failed_inferences: int = 0
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    errors_by_model: dict[str, int] = field(default_factory=dict)


class InferencePipeline:
    """Coordinate data-triggered inference, caching, and signal publication."""

    def __init__(
        self,
        model_registry: SupportingModelRegistry,
        strategy_registry: CustomStrategyRegistry,
        data_router: DataRouter,
        cache_config: dict[str, object] | None = None,
    ) -> None:
        """Initialize pipeline dependencies and default runtime configuration."""

        self._model_registry = model_registry
        self._strategy_registry = strategy_registry
        self._data_router = data_router

        cache_cfg = cache_config or {}
        self._cache = SignalCache(
            max_age_seconds=float(cache_cfg.get("max_age_seconds", 300.0)),
            max_signals_per_model=int(cache_cfg.get("max_signals_per_model", 100)),
        )

        self._config: dict[str, object] = {
            "max_workers": 4,
            "default_timeout_seconds": 5.0,
            "model_timeouts": {},
            "prune_every_n_results": 50,
        }
        self._executor: InferenceExecutor | None = None
        self._running = False

        self._metrics = InferenceMetrics()
        self._metrics_lock = RLock()
        self._state_lock = RLock()

        self._listener_callbacks: list[Callable[[str, ModelSignal], None]] = []
        self._router_handler_ids: list[str] = []
        self._completed_since_prune = 0
        self._configured_news_sentiment: dict[str, tuple[NewsDataSource, list[str]]] = (
            {}
        )
        self._active_news_subscriptions: dict[str, tuple[NewsDataSource, int]] = {}

    def configure(self, config: dict[str, object]) -> None:
        """Update runtime configuration for executor and timeouts."""

        with self._state_lock:
            self._config.update(config)

    def start(self) -> None:
        """Start pipeline execution by subscribing handlers and creating workers."""

        with self._state_lock:
            if self._running:
                return

            self._executor = InferenceExecutor(
                max_workers=int(self._config["max_workers"]),
                default_timeout=float(self._config["default_timeout_seconds"]),
            )

            for data_type in self._collect_subscription_data_types():
                handler_id = self._data_router.register_callback(
                    self._on_data_received,
                    data_type,
                )
                self._router_handler_ids.append(handler_id)

            self._running = True
            self._activate_configured_news_subscriptions()

    def stop(self) -> None:
        """Stop processing and cleanly release subscriptions and workers."""

        with self._state_lock:
            if not self._running:
                return

            for handler_id in self._router_handler_ids:
                self._data_router.unregister_callback(handler_id)
            self._router_handler_ids.clear()

            self._stop_news_subscriptions()

            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None

            self._running = False

    def is_running(self) -> bool:
        """Return whether the inference pipeline is currently active."""

        with self._state_lock:
            return self._running

    def get_latest_signal(self, model_id: str) -> ModelSignal | None:
        """Return latest cached signal for one model or strategy."""

        return self._cache.get_latest(model_id)

    def get_all_latest_signals(self) -> dict[str, ModelSignal]:
        """Return latest signal per model from cache."""

        return self._cache.get_all_latest()

    def get_signal_before(
        self,
        model_id: str,
        timestamp: datetime,
    ) -> ModelSignal | None:
        """Return latest cached signal at or before provided timestamp."""

        return self._cache.get_latest_before(model_id, timestamp)

    def submit_record(self, record: DataRecord) -> list[Future[InferenceResult]]:
        """Submit one data record directly to matching supporting providers.

        This is used by live trading paths where broker callbacks already have a
        normalized market record and do not need a ``DataSource`` streaming loop.
        The same completion callbacks, cache updates, listeners, and metrics are
        used as router-driven inference.
        """

        with self._state_lock:
            if not self._running or self._executor is None:
                return []
            executor = self._executor

        providers = self._find_models_for_data_type(record.data_type)
        if not providers:
            return []

        futures: list[Future[InferenceResult]] = []
        for provider in providers:
            model_id = self._resolve_provider_id(provider)
            task = InferenceTask(
                task_id=str(uuid4()),
                model_id=model_id,
                data=record,
                submitted_at=datetime.now(tz=UTC),
                timeout_seconds=self._get_timeout(model_id),
            )
            future = executor.submit(task, provider)
            future.add_done_callback(self._on_inference_complete)
            futures.append(future)

        return futures

    def get_metrics(self) -> InferenceMetrics:
        """Return a snapshot copy of current inference metrics."""

        with self._metrics_lock:
            return InferenceMetrics(
                total_inferences=self._metrics.total_inferences,
                successful_inferences=self._metrics.successful_inferences,
                failed_inferences=self._metrics.failed_inferences,
                total_latency_ms=self._metrics.total_latency_ms,
                max_latency_ms=self._metrics.max_latency_ms,
                errors_by_model=dict(self._metrics.errors_by_model),
            )

    def reset_metrics(self) -> None:
        """Reset all collected inference metrics."""

        with self._metrics_lock:
            self._metrics = InferenceMetrics()

    def add_signal_listener(self, callback: Callable[[str, ModelSignal], None]) -> None:
        """Register callback invoked after successful signal inference."""

        with self._state_lock:
            self._listener_callbacks.append(callback)

    def configure_news_sentiment(
        self,
        news_source: NewsDataSource,
        sentiment_model_id: str,
        symbols: list[str],
    ) -> None:
        """Configure direct news provider subscription for one sentiment model.

        This enables end-to-end news -> sentiment inference flow even when news
        records arrive outside router-managed streams.
        """

        model_id = sentiment_model_id.strip()
        if not model_id:
            raise ValueError("sentiment_model_id must be non-empty")

        normalized_symbols = sorted(
            {value.strip().upper() for value in symbols if value.strip()}
        )
        if not normalized_symbols:
            raise ValueError("symbols must include at least one non-empty symbol")

        entry = self._model_registry.get(model_id)
        if entry is None:
            raise ValueError(f"Sentiment model '{model_id}' is not registered")
        if DataType.NEWS_TEXT not in entry.config.input_data_types:
            logger.warning(
                "Configured model %s for direct news sentiment but it does not "
                "declare NEWS_TEXT in input_data_types",
                model_id,
            )

        with self._state_lock:
            self._configured_news_sentiment[model_id] = (
                news_source,
                normalized_symbols,
            )
            running = self._running

        if running:
            self._activate_news_subscription(
                model_id=model_id,
                news_source=news_source,
                symbols=normalized_symbols,
            )

    def _activate_configured_news_subscriptions(self) -> None:
        """Activate all configured direct news subscriptions."""

        configured = list(self._configured_news_sentiment.items())
        for model_id, (news_source, symbols) in configured:
            self._activate_news_subscription(
                model_id=model_id,
                news_source=news_source,
                symbols=symbols,
            )

    def _activate_news_subscription(
        self,
        model_id: str,
        news_source: NewsDataSource,
        symbols: list[str],
    ) -> None:
        """Activate direct news callback subscription for a model id."""

        with self._state_lock:
            if model_id in self._active_news_subscriptions:
                return

        if not news_source.is_connected:
            news_source.connect()

        def _on_news(news: NewsRecord) -> None:
            self._on_news_received(
                model_id=model_id,
                news_source=news_source,
                news=news,
            )

        subscription_id = news_source.subscribe_news(symbols=symbols, callback=_on_news)
        with self._state_lock:
            self._active_news_subscriptions[model_id] = (news_source, subscription_id)

    def _stop_news_subscriptions(self) -> None:
        """Stop all active direct news subscriptions."""

        with self._state_lock:
            subscriptions = list(self._active_news_subscriptions.items())
            self._active_news_subscriptions.clear()

        for model_id, (news_source, subscription_id) in subscriptions:
            try:
                news_source.unsubscribe_news(subscription_id)
            except Exception:  # pragma: no cover - defensive callback handling
                logger.exception(
                    "Failed to stop news subscription for model %s",
                    model_id,
                )

    def _on_news_received(
        self,
        model_id: str,
        news_source: NewsDataSource,
        news: NewsRecord,
    ) -> None:
        """Submit one direct news record to a configured sentiment model."""

        with self._state_lock:
            if not self._running or self._executor is None:
                return
            executor = self._executor

        entry = self._model_registry.get(model_id)
        if entry is None:
            logger.warning(
                "Skipping direct news inference for unknown model %s",
                model_id,
            )
            return
        if entry.trainer is None:
            logger.warning(
                "Skipping direct news inference for unloaded model %s",
                model_id,
            )
            return

        record = news_source.news_to_record(news)
        task = InferenceTask(
            task_id=str(uuid4()),
            model_id=model_id,
            data=record,
            submitted_at=datetime.now(tz=UTC),
            timeout_seconds=self._get_timeout(model_id),
        )
        future = executor.submit(task, entry)
        future.add_done_callback(self._on_inference_complete)

    def _collect_subscription_data_types(self) -> list[DataType]:
        """Collect data types consumed by registered models and strategies."""

        types: set[DataType] = set()

        for entry in self._model_registry.get_all():
            types.update(entry.config.input_data_types)

        for strategy in self._strategy_registry.get_all():
            metadata = getattr(strategy.strategy_class, "_strategy_metadata", {})
            raw_types = metadata.get("input_data_types", [DataType.MARKET_BAR])
            if isinstance(raw_types, list):
                for value in raw_types:
                    if isinstance(value, DataType):
                        types.add(value)
                    elif isinstance(value, str):
                        types.add(DataType(value))

        return sorted(types, key=lambda value: value.value)

    def _find_models_for_data_type(
        self,
        data_type: DataType,
    ) -> list[ModelEntry | StrategyEntry]:
        """Return providers that should run for the incoming data type."""

        providers: list[ModelEntry | StrategyEntry] = []

        for entry in self._model_registry.get_all():
            if data_type in entry.config.input_data_types and entry.trainer is not None:
                providers.append(entry)

        for strategy in self._strategy_registry.get_all():
            metadata = getattr(strategy.strategy_class, "_strategy_metadata", {})
            raw_types = metadata.get("input_data_types", [DataType.MARKET_BAR])
            allowed: set[DataType] = set()
            for value in raw_types:
                if isinstance(value, DataType):
                    allowed.add(value)
                elif isinstance(value, str):
                    allowed.add(DataType(value))
            if data_type in allowed:
                providers.append(strategy)

        return providers

    def _on_data_received(self, record: DataRecord) -> None:
        """Handle router data callback by dispatching provider inference tasks."""

        self.submit_record(record)

    def _on_inference_complete(self, future: Future[InferenceResult]) -> None:
        """Handle completed inference futures and update cache and metrics."""

        result = future.result()
        self._update_metrics(result)

        if result.success and result.signal is not None:
            self._cache.put(result.model_id, result.signal)
            self._notify_listeners(result.model_id, result.signal)
        else:
            logger.error(
                "Inference failed for model %s: %s",
                result.model_id,
                result.error_message,
            )

        prune_every = int(self._config.get("prune_every_n_results", 50))
        if prune_every > 0:
            self._completed_since_prune += 1
            if self._completed_since_prune >= prune_every:
                self._cache.prune_old()
                self._completed_since_prune = 0

    def _update_metrics(self, result: InferenceResult) -> None:
        """Update aggregate metrics from one inference completion result."""

        with self._metrics_lock:
            self._metrics.total_inferences += 1
            self._metrics.total_latency_ms += result.inference_time_ms
            self._metrics.max_latency_ms = max(
                self._metrics.max_latency_ms,
                result.inference_time_ms,
            )

            if result.success:
                self._metrics.successful_inferences += 1
            else:
                self._metrics.failed_inferences += 1
                self._metrics.errors_by_model[result.model_id] = (
                    self._metrics.errors_by_model.get(result.model_id, 0) + 1
                )

    def _notify_listeners(self, model_id: str, signal: ModelSignal) -> None:
        """Notify registered listeners when a new signal is available."""

        callbacks: list[Callable[[str, ModelSignal], None]]
        with self._state_lock:
            callbacks = list(self._listener_callbacks)

        for callback in callbacks:
            try:
                callback(model_id, signal)
            except Exception:  # pragma: no cover - defensive callback handling
                logger.exception("Signal listener failed for model %s", model_id)

    def _resolve_provider_id(self, provider: ModelEntry | StrategyEntry) -> str:
        """Return canonical model identifier for provider entry."""

        if isinstance(provider, StrategyEntry):
            return provider.strategy_id
        return provider.config.model_id

    def _get_timeout(self, model_id: str) -> float:
        """Resolve timeout for a given model from per-model or default config."""

        timeouts = self._config.get("model_timeouts", {})
        if isinstance(timeouts, dict) and model_id in timeouts:
            return float(timeouts[model_id])
        return float(self._config["default_timeout_seconds"])
