"""Streaming data router that connects sources to pipeline consumers."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Event, RLock, Thread
from typing import Protocol
from uuid import uuid4

from algotrading.src.data_pipeline.routing.buffer import (
    BufferConfig,
    MultiFrequencyBuffer,
)
from algotrading.src.data_pipeline.routing.subscription import (
    Subscription,
    SubscriptionManager,
)
from algotrading.src.data_pipeline.sources.base import DataSource
from algotrading.src.data_pipeline.state.state_manager import StateManager
from algotrading.src.data_pipeline.types import DataRecord, DataType


class DataRouterError(RuntimeError):
    """Raised when the router configuration or lifecycle is invalid."""


class DataHandler(Protocol):
    """Protocol for user-defined record handlers."""

    def __call__(self, record: DataRecord) -> None:
        """Handle one streamed data record."""


@dataclass(slots=True)
class RouteConfig:
    """Route definition for one data source.

    Args:
        source: Source instance to route from.
        data_types: Data types to stream from the source.
        symbols: Symbols to stream for each data type.
        targets: Routing targets. Supported values are
            ``state_manager``, ``buffer``, and ``callback``.
    """

    source: DataSource
    data_types: list[DataType]
    symbols: list[str]
    targets: list[str]

    def __post_init__(self) -> None:
        """Validate route configuration."""

        if not self.data_types:
            raise ValueError("RouteConfig.data_types must not be empty")
        if not self.symbols:
            raise ValueError("RouteConfig.symbols must not be empty")

        allowed_targets = {"state_manager", "buffer", "callback"}
        unknown = [target for target in self.targets if target not in allowed_targets]
        if unknown:
            raise ValueError(f"Unsupported route targets: {unknown}")


@dataclass(slots=True)
class RouterConfig:
    """Runtime settings for ``DataRouter``.

    Args:
        routes: Source route configurations.
        buffer_config: Optional auto-created buffer configuration.
        retry_attempts: Number of retries after a source stream failure.
        retry_delay_seconds: Delay between retries.
        stop_join_timeout_seconds: Maximum wait for stream worker shutdown.
    """

    routes: list[RouteConfig] = field(default_factory=list)
    buffer_config: BufferConfig | None = None
    retry_attempts: int = 1
    retry_delay_seconds: float = 0.5
    stop_join_timeout_seconds: float = 2.0

    def __post_init__(self) -> None:
        """Validate router config bounds."""

        if self.retry_attempts < 0:
            raise ValueError("retry_attempts must be >= 0")
        if self.retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must be >= 0")
        if self.stop_join_timeout_seconds <= 0:
            raise ValueError("stop_join_timeout_seconds must be > 0")


class DataRouter:
    """Coordinate streaming records from ``DataSource`` to router targets."""

    def __init__(self, config: RouterConfig | None = None) -> None:
        """Initialise router state and optional configured routes."""

        self._logger = logging.getLogger(__name__)
        self._lock = RLock()
        self._config = config or RouterConfig()

        self._subscription_manager = SubscriptionManager()
        self._sources: dict[str, DataSource] = {}
        self._routes: dict[str, RouteConfig] = {}

        self._state_managers: dict[DataType, StateManager] = {}
        self._buffer: MultiFrequencyBuffer | None = None
        self._callbacks: dict[DataType, dict[str, DataHandler]] = {}
        self._error_callbacks: dict[str, Callable[[str, Exception], None]] = {}

        self._threads: dict[str, Thread] = {}
        self._stop_events: dict[str, Event] = {}

        if self._config.buffer_config is not None:
            self._buffer = MultiFrequencyBuffer(config=self._config.buffer_config)

        for route in self._config.routes:
            self.add_source(route.source, config=route)

    def register_state_manager(
        self,
        manager: StateManager,
        data_type: DataType = DataType.MARKET_BAR,
    ) -> None:
        """Register a state manager target for one data type."""

        with self._lock:
            self._state_managers[data_type] = manager

    def register_buffer(self, buffer: MultiFrequencyBuffer) -> None:
        """Register an externally managed multi-frequency buffer."""

        with self._lock:
            self._buffer = buffer

    def register_callback(
        self,
        callback: DataHandler,
        data_type: DataType,
    ) -> str:
        """Register a callback for records of ``data_type``.

        Returns:
            Handler identifier used for unregistration.
        """

        handler_id = f"handler_{uuid4().hex}"
        with self._lock:
            self._callbacks.setdefault(data_type, {})[handler_id] = callback
        return handler_id

    def unregister_callback(self, handler_id: str) -> None:
        """Unregister a previously registered callback by identifier."""

        with self._lock:
            for handlers in self._callbacks.values():
                if handler_id in handlers:
                    del handlers[handler_id]
                    return

    def register_error_callback(
        self,
        callback: Callable[[str, Exception], None],
    ) -> str:
        """Register a callback invoked on source streaming errors."""

        callback_id = f"error_{uuid4().hex}"
        with self._lock:
            self._error_callbacks[callback_id] = callback
        return callback_id

    def unregister_error_callback(self, callback_id: str) -> None:
        """Unregister an error callback."""

        with self._lock:
            self._error_callbacks.pop(callback_id, None)

    def add_source(self, source: DataSource, config: RouteConfig | None = None) -> None:
        """Add a source and optional explicit route configuration."""

        route = config or self._default_route_for_source(source)
        if route.source.source_id != source.source_id:
            raise ValueError("RouteConfig source_id does not match add_source source")

        with self._lock:
            self._sources[source.source_id] = source
            self._routes[source.source_id] = route

    def remove_source(self, source_id: str) -> None:
        """Stop and remove source plus associated route configuration."""

        self.stop_streaming(source_id=source_id)
        with self._lock:
            source = self._sources.pop(source_id, None)
            self._routes.pop(source_id, None)

        if source is not None and source.is_connected:
            source.disconnect()

    def get_sources(self) -> list[DataSource]:
        """Return all currently registered sources."""

        with self._lock:
            return list(self._sources.values())

    def start_streaming(self, source_id: str | None = None) -> None:
        """Start streaming for one source or all registered sources."""

        source_ids = self._resolve_source_ids(source_id)
        for current_source_id in source_ids:
            source = self._sources[current_source_id]
            route = self._routes[current_source_id]

            if not source.is_connected:
                source.connect()

            if self._subscription_manager.list_by_source(current_source_id):
                continue

            for data_type in route.data_types:
                for symbol in route.symbols:
                    subscription = self._subscription_manager.create(
                        source=source,
                        data_type=data_type,
                        symbol=symbol,
                        handlers=self._get_handlers(data_type),
                    )
                    stop_event = Event()
                    worker = Thread(
                        target=self._stream_worker,
                        kwargs={
                            "subscription": subscription,
                            "targets": list(route.targets),
                            "stop_event": stop_event,
                        },
                        name=f"router-{current_source_id}-{data_type.value}-{symbol or 'default'}",
                        daemon=True,
                    )
                    with self._lock:
                        self._threads[subscription.id] = worker
                        self._stop_events[subscription.id] = stop_event
                    worker.start()

    def stop_streaming(self, source_id: str | None = None) -> None:
        """Stop active streaming workers for one source or all sources."""

        source_ids = self._resolve_source_ids(source_id)
        active_subscriptions = [
            subscription
            for subscription in self._subscription_manager.list_active()
            if subscription.source.source_id in source_ids
        ]

        for subscription in active_subscriptions:
            stop_event = self._stop_events.get(subscription.id)
            if stop_event is not None:
                stop_event.set()
            self._subscription_manager.cancel(subscription.id)

        for subscription in active_subscriptions:
            thread = self._threads.get(subscription.id)
            if thread is None:
                continue
            thread.join(timeout=self._config.stop_join_timeout_seconds)
            with self._lock:
                self._threads.pop(subscription.id, None)
                self._stop_events.pop(subscription.id, None)

    def is_streaming(self, source_id: str | None = None) -> bool:
        """Return whether streaming is active for one source or any source."""

        if source_id is None:
            return len(self._subscription_manager.list_active()) > 0

        return len(self._subscription_manager.list_by_source(source_id)) > 0

    def on_source_error(self, source_id: str, error: Exception) -> None:
        """Handle source stream failures and notify registered callbacks."""

        self._logger.error("Source stream error from %s: %s", source_id, error)
        callbacks = list(self._error_callbacks.values())
        for callback in callbacks:
            try:
                callback(source_id, error)
            except Exception as callback_error:  # pragma: no cover - defensive logging
                self._logger.error(
                    "Error callback failed for source %s: %s",
                    source_id,
                    callback_error,
                )

    def start(self) -> None:
        """Connect all registered sources and begin streaming."""

        self.start_streaming()

    def stop(self) -> None:
        """Stop all streaming workers and disconnect sources."""

        self.stop_streaming()
        for source in self.get_sources():
            if source.is_connected:
                source.disconnect()

    def __enter__(self) -> DataRouter:
        """Start routing when entering context manager."""

        self.start()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Stop routing when exiting context manager."""

        self.stop()

    def _stream_worker(
        self,
        subscription: Subscription,
        targets: list[str],
        stop_event: Event,
    ) -> None:
        """Run one source stream and route records until stopped."""

        attempts = 0
        try:
            while not stop_event.is_set():
                try:
                    for record in subscription.source.fetch_stream(
                        symbol=subscription.symbol,
                        data_type=subscription.data_type,
                    ):
                        if stop_event.is_set():
                            break
                        if not subscription.source.validate_record(record):
                            self._logger.debug(
                                "Skipping invalid record from %s",
                                subscription.source.source_id,
                            )
                            continue
                        self._route_record(record, targets=targets)
                    break
                except Exception as stream_error:
                    attempts += 1
                    self.on_source_error(subscription.source.source_id, stream_error)
                    if attempts > self._config.retry_attempts or stop_event.is_set():
                        break
                    time.sleep(self._config.retry_delay_seconds)
        finally:
            self._subscription_manager.cancel(subscription.id)
            with self._lock:
                self._threads.pop(subscription.id, None)
                self._stop_events.pop(subscription.id, None)

    def _route_record(self, record: DataRecord, targets: list[str]) -> None:
        """Route one record to configured destinations."""

        if "buffer" in targets and self._buffer is not None:
            self._buffer.put(record)

        if "state_manager" in targets:
            manager = self._state_managers.get(record.data_type)
            if manager is not None:
                manager.append_record(record)

        if "callback" in targets:
            handlers = self._get_handlers(record.data_type)
            for handler in handlers:
                handler(record)

    def _get_handlers(self, data_type: DataType) -> list[DataHandler]:
        """Return callbacks currently registered for ``data_type``."""

        with self._lock:
            return list(self._callbacks.get(data_type, {}).values())

    def _resolve_source_ids(self, source_id: str | None) -> list[str]:
        """Resolve source identifiers for target operation."""

        with self._lock:
            if source_id is None:
                return list(self._sources.keys())
            if source_id not in self._sources:
                raise DataRouterError(f"Unknown source id: {source_id}")
            return [source_id]

    def _default_route_for_source(self, source: DataSource) -> RouteConfig:
        """Build a default route when explicit config is not provided."""

        data_types = list(source.metadata.supported_types)
        symbols = source.get_available_symbols() if source.is_connected else []
        if not symbols:
            symbols = [""]

        targets = ["callback"]
        if DataType.MARKET_BAR in self._state_managers:
            targets.append("state_manager")
        if self._buffer is not None:
            targets.append("buffer")

        return RouteConfig(
            source=source,
            data_types=data_types,
            symbols=symbols,
            targets=targets,
        )
