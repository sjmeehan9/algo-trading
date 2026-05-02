"""Real-time market-data streaming for the Alpaca adapter."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import defaultdict, deque
from datetime import UTC, datetime
from typing import Callable

from algotrading.src.broker.adapters.alpaca.auth import AlpacaAuth
from algotrading.src.broker.adapters.alpaca.models import (
    alpaca_bar_to_bar_data,
    contract_to_alpaca_symbol,
)
from algotrading.src.broker.models import BarData, ContractSpec

logger = logging.getLogger(__name__)


class AlpacaStreamClient:
    """Threaded wrapper around Alpaca's stock bar WebSocket stream."""

    HEARTBEAT_STALE_SECONDS = 90.0
    RECONNECT_DELAY_SECONDS = 2.0

    def __init__(self, auth: AlpacaAuth, stream_client: object | None = None) -> None:
        """Initialize streaming state with optional injected SDK stream."""

        self.auth = auth
        self._stream_client = stream_client
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._reconnecting = False
        self._subscription_seed = 1
        self._subscriptions: dict[int, tuple[str, Callable[[BarData], None]]] = {}
        self._symbol_subscriptions: dict[str, set[int]] = defaultdict(set)
        self._pending_replay: deque[tuple[str, BarData]] = deque(maxlen=1_000)
        self._last_message_at: datetime | None = None

    @property
    def stream_client(self) -> object:
        """Return the underlying stream client, creating it lazily."""

        if self._stream_client is None:
            self._stream_client = self.auth.get_stream_client()
        return self._stream_client

    def subscribe_bars(
        self,
        contract: ContractSpec,
        bar_size: int,
        callback: Callable[[BarData], None],
    ) -> int:
        """Subscribe to Alpaca stock bars and return a local subscription id."""

        if bar_size <= 0:
            raise ValueError("bar_size must be greater than zero")
        symbol = contract_to_alpaca_symbol(contract)
        with self._lock:
            subscription_id = self._subscription_seed
            self._subscription_seed += 1
            is_first_symbol_subscription = not self._symbol_subscriptions[symbol]
            self._subscriptions[subscription_id] = (symbol, callback)
            self._symbol_subscriptions[symbol].add(subscription_id)

        if is_first_symbol_subscription:
            self.stream_client.subscribe_bars(self._handle_bar, symbol)
        self.start()
        return subscription_id

    def unsubscribe(self, subscription_id: int) -> None:
        """Cancel a local bar subscription."""

        with self._lock:
            item = self._subscriptions.pop(subscription_id, None)
            if item is None:
                return
            symbol, _callback = item
            subscription_ids = self._symbol_subscriptions[symbol]
            subscription_ids.discard(subscription_id)
            should_unsubscribe_symbol = not subscription_ids
            if should_unsubscribe_symbol:
                self._symbol_subscriptions.pop(symbol, None)

        if should_unsubscribe_symbol and hasattr(
            self.stream_client, "unsubscribe_bars"
        ):
            self.stream_client.unsubscribe_bars(symbol)

    def start(self) -> None:
        """Start the stream loop in a daemon thread if needed."""

        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._running = True
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stop the stream client and clear runtime state."""

        with self._lock:
            self._running = False
            stream_client = self._stream_client

        if stream_client is not None and hasattr(stream_client, "stop"):
            result = stream_client.stop()
            if asyncio.iscoroutine(result):
                try:
                    asyncio.run(result)
                except RuntimeError:
                    logger.debug("Skipping async stream stop inside active event loop")

    def get_status(self) -> dict[str, object]:
        """Return stream health and subscription metadata."""

        with self._lock:
            last_message = self._last_message_at
            subscription_count = len(self._subscriptions)
            symbol_count = len(self._symbol_subscriptions)

        age_seconds = None
        healthy = False
        if last_message is not None:
            age_seconds = (datetime.now(tz=UTC) - last_message).total_seconds()
            healthy = age_seconds <= self.HEARTBEAT_STALE_SECONDS
        return {
            "running": self._running,
            "healthy": healthy,
            "last_message_at": last_message.isoformat() if last_message else None,
            "last_message_age_seconds": age_seconds,
            "subscription_count": subscription_count,
            "symbol_count": symbol_count,
        }

    async def _handle_bar(self, bar: object) -> None:
        """Handle an Alpaca SDK bar callback."""

        bar_data = alpaca_bar_to_bar_data(bar)
        symbol = str(getattr(bar, "symbol", "") or "").upper()
        if not symbol and isinstance(bar, dict):
            symbol = str(bar.get("symbol", "") or "").upper()

        with self._lock:
            self._last_message_at = datetime.now(tz=UTC)
            callbacks = [
                callback
                for subscribed_symbol, callback in self._subscriptions.values()
                if not symbol or subscribed_symbol == symbol
            ]
            if self._reconnecting:
                self._pending_replay.append((symbol, bar_data))
                return

        for callback in callbacks:
            callback(bar_data)

    def _run_loop(self) -> None:
        while self._should_run():
            try:
                self._reconnecting = False
                self.stream_client.run()
                with self._lock:
                    self._running = False
                return
            except Exception as exc:
                if not self._should_run():
                    return
                logger.warning("Alpaca stream disconnected: %s", exc)
                self._reconnecting = True
                time.sleep(self.RECONNECT_DELAY_SECONDS)
                self._stream_client = self.auth.get_stream_client()
                self._resubscribe_symbols()
                self._replay_pending()

    def _resubscribe_symbols(self) -> None:
        with self._lock:
            symbols = list(self._symbol_subscriptions)
        for symbol in symbols:
            self.stream_client.subscribe_bars(self._handle_bar, symbol)

    def _replay_pending(self) -> None:
        while True:
            with self._lock:
                if not self._pending_replay:
                    return
                symbol, bar_data = self._pending_replay.popleft()
                callbacks = [
                    callback
                    for subscribed_symbol, callback in self._subscriptions.values()
                    if not symbol or subscribed_symbol == symbol
                ]
            for callback in callbacks:
                callback(bar_data)

    def _should_run(self) -> bool:
        with self._lock:
            return self._running


__all__ = ["AlpacaStreamClient"]
