"""Subscription tracking primitives for streaming data routing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock
from uuid import uuid4

from algotrading.src.data_pipeline.sources.base import DataSource
from algotrading.src.data_pipeline.types import DataRecord, DataType


@dataclass(slots=True)
class Subscription:
    """Track one active source/data-type/symbol subscription.

    Args:
        id: Unique subscription identifier.
        source: Data source associated with the stream.
        data_type: Data type streamed through the subscription.
        symbol: Symbol requested from the source stream.
        handlers: Routing handlers invoked for each received record.
        is_active: Whether the subscription is currently active.
        created_at: UTC timestamp when the subscription was created.
    """

    id: str
    source: DataSource
    data_type: DataType
    symbol: str
    handlers: list[Callable[[DataRecord], None]]
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class SubscriptionManager:
    """Thread-safe registry for active streaming subscriptions."""

    def __init__(self) -> None:
        """Initialise internal subscription storage."""

        self._subscriptions: dict[str, Subscription] = {}
        self._lock = Lock()

    def create(
        self,
        source: DataSource,
        data_type: DataType,
        symbol: str,
        handlers: list[Callable[[DataRecord], None]],
    ) -> Subscription:
        """Create and store a new active subscription.

        Args:
            source: Data source for the stream.
            data_type: Data type being streamed.
            symbol: Symbol for the stream.
            handlers: Handlers invoked per received record.

        Returns:
            Created subscription instance.
        """

        subscription = Subscription(
            id=f"sub_{uuid4().hex}",
            source=source,
            data_type=data_type,
            symbol=symbol,
            handlers=list(handlers),
        )
        with self._lock:
            self._subscriptions[subscription.id] = subscription
        return subscription

    def get(self, subscription_id: str) -> Subscription | None:
        """Get a subscription by identifier."""

        with self._lock:
            return self._subscriptions.get(subscription_id)

    def cancel(self, subscription_id: str) -> bool:
        """Mark a subscription inactive and remove it from active registry.

        Args:
            subscription_id: Subscription identifier.

        Returns:
            True when the subscription existed and was cancelled.
        """

        with self._lock:
            subscription = self._subscriptions.get(subscription_id)
            if subscription is None:
                return False
            subscription.is_active = False
            del self._subscriptions[subscription_id]
            return True

    def list_active(self) -> list[Subscription]:
        """Return all active subscriptions."""

        with self._lock:
            return [sub for sub in self._subscriptions.values() if sub.is_active]

    def list_by_source(self, source_id: str) -> list[Subscription]:
        """Return active subscriptions belonging to ``source_id``."""

        with self._lock:
            return [
                sub
                for sub in self._subscriptions.values()
                if sub.is_active and sub.source.source_id == source_id
            ]
