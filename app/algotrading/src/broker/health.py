"""Broker health-check utilities."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta

from algotrading.src.broker.adapter import BrokerAdapter

logger = logging.getLogger(__name__)


class BrokerHealth:
    """Cached health probe for a broker adapter.

    Args:
        broker: Broker adapter to probe.
        cache_ttl_seconds: Number of seconds to cache health results.
        require_account_snapshot: When True, ``get_account_info()`` must also
            succeed for the broker to be considered healthy.
    """

    def __init__(
        self,
        broker: BrokerAdapter,
        *,
        cache_ttl_seconds: float = 30.0,
        require_account_snapshot: bool = True,
    ) -> None:
        """Initialize the health checker."""

        self.broker = broker
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self.require_account_snapshot = require_account_snapshot
        self._last_check: datetime | None = None
        self._last_status = False
        self._last_message = "not checked"
        self._last_latency_ms: float | None = None

    def check(self, force: bool = False) -> bool:
        """Check broker health, using a short cache by default.

        Args:
            force: When True, bypass the cached result.

        Returns:
            True when the broker is connected and required probes pass.
        """

        now = datetime.now(tz=UTC)
        if (
            not force
            and self._last_check is not None
            and now - self._last_check < self.cache_ttl
        ):
            return self._last_status

        start = time.perf_counter()
        try:
            if not self.broker.is_connected():
                self._last_status = False
                self._last_message = "broker adapter is disconnected"
                return self._last_status
            if self.require_account_snapshot:
                self.broker.get_account_info()
            self._last_status = True
            self._last_message = "healthy"
            return self._last_status
        except Exception as exc:
            self._last_status = False
            self._last_message = str(exc)
            logger.warning("Broker health check failed: %s", type(self.broker).__name__)
            return self._last_status
        finally:
            self._last_check = datetime.now(tz=UTC)
            self._last_latency_ms = (time.perf_counter() - start) * 1000

    def get_status(self) -> dict[str, object]:
        """Return detailed health status for the broker."""

        healthy = self.check()
        return {
            "healthy": healthy,
            "last_check": (
                self._last_check.isoformat() if self._last_check is not None else None
            ),
            "broker_type": type(self.broker).__name__,
            "message": self._last_message,
            "latency_ms": self._last_latency_ms,
        }


__all__ = ["BrokerHealth"]
