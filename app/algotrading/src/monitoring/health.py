"""Application health checks for production operations."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from algotrading.src.broker import BrokerRegistry


class HealthStatus(str, Enum):
    """Possible component and aggregate health states."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass(slots=True)
class ComponentHealth:
    """Health result for one application component."""

    name: str
    status: HealthStatus
    message: str | None = None
    latency_ms: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialize the component health result."""

        payload: dict[str, object] = {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": self.latency_ms,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


class HealthChecker:
    """Run broker, persistence, and session health checks."""

    def __init__(
        self,
        broker_registry: BrokerRegistry | None = None,
        session_manager_provider: Callable[[], object | None] | None = None,
        persistence_path: str | Path | None = None,
    ) -> None:
        """Initialize health check dependencies.

        Args:
            broker_registry: Registry whose active broker should be probed.
            session_manager_provider: Callable returning the current session manager.
            persistence_path: Optional storage path to probe when no manager exists.
        """

        self.broker_registry = broker_registry
        self.session_manager_provider = session_manager_provider or (lambda: None)
        self.persistence_path = Path(persistence_path) if persistence_path else None

    async def check_all(self) -> dict[str, object]:
        """Run all configured health checks and return an aggregate payload."""

        components = [
            await self._check_broker(),
            await self._check_persistence(),
            await self._check_sessions(),
        ]
        overall = self._overall_status(components)
        return {
            "status": overall.value,
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "components": [component.to_dict() for component in components],
        }

    async def _check_broker(self) -> ComponentHealth:
        start = time.perf_counter()
        if self.broker_registry is None:
            return ComponentHealth(
                name="broker",
                status=HealthStatus.HEALTHY,
                message="broker registry not initialized",
                latency_ms=_elapsed_ms(start),
            )

        broker = self.broker_registry.get_active_broker()
        broker_name = self.broker_registry.get_active_broker_name()
        if broker is None:
            return ComponentHealth(
                name="broker",
                status=HealthStatus.HEALTHY,
                message="no active broker connection",
                latency_ms=_elapsed_ms(start),
                metadata={"active": False},
            )

        try:
            if not broker.is_connected():
                return ComponentHealth(
                    name="broker",
                    status=HealthStatus.UNHEALTHY,
                    message="active broker is disconnected",
                    latency_ms=_elapsed_ms(start),
                    metadata={"broker": broker_name or type(broker).__name__},
                )
            broker.get_account_info()
        except Exception as exc:
            return ComponentHealth(
                name="broker",
                status=HealthStatus.UNHEALTHY,
                message=str(exc),
                latency_ms=_elapsed_ms(start),
                metadata={"broker": broker_name or type(broker).__name__},
            )

        return ComponentHealth(
            name="broker",
            status=HealthStatus.HEALTHY,
            message="connected",
            latency_ms=_elapsed_ms(start),
            metadata={"broker": broker_name or type(broker).__name__, "active": True},
        )

    async def _check_persistence(self) -> ComponentHealth:
        start = time.perf_counter()
        storage_path = self._resolve_persistence_path()
        if storage_path is None:
            return ComponentHealth(
                name="persistence",
                status=HealthStatus.HEALTHY,
                message="persistence not initialized",
                latency_ms=_elapsed_ms(start),
            )

        try:
            await asyncio.to_thread(_probe_storage_path, storage_path)
        except Exception as exc:
            return ComponentHealth(
                name="persistence",
                status=HealthStatus.UNHEALTHY,
                message=str(exc),
                latency_ms=_elapsed_ms(start),
                metadata={"path": str(storage_path)},
            )

        return ComponentHealth(
            name="persistence",
            status=HealthStatus.HEALTHY,
            message="storage writable",
            latency_ms=_elapsed_ms(start),
            metadata={"path": str(storage_path)},
        )

    async def _check_sessions(self) -> ComponentHealth:
        start = time.perf_counter()
        manager = self.session_manager_provider()
        if manager is None:
            return ComponentHealth(
                name="sessions",
                status=HealthStatus.HEALTHY,
                message="session manager not initialized",
                latency_ms=_elapsed_ms(start),
            )

        sessions = await self._collect_sessions(manager)
        running = sum(
            1 for session in sessions if _session_status(session) == "running"
        )
        paused = sum(1 for session in sessions if _session_status(session) == "paused")
        errored = sum(1 for session in sessions if _session_status(session) == "error")
        total_errors = sum(_session_error_count(session) for session in sessions)

        status = HealthStatus.DEGRADED if errored else HealthStatus.HEALTHY
        message = (
            f"{errored} sessions in error state"
            if errored
            else f"{running} sessions running, {paused} paused"
        )
        return ComponentHealth(
            name="sessions",
            status=status,
            message=message,
            latency_ms=_elapsed_ms(start),
            metadata={
                "total": len(sessions),
                "running": running,
                "paused": paused,
                "errored": errored,
                "error_count": total_errors,
            },
        )

    async def _collect_sessions(self, manager: object) -> list[object]:
        session_map = getattr(manager, "_sessions", None)
        if isinstance(session_map, dict):
            return list(session_map.values())

        list_sessions = getattr(manager, "list_sessions", None)
        if callable(list_sessions):
            sessions = list_sessions()
            if inspect.isawaitable(sessions):
                sessions = await sessions
            return list(sessions)

        return []

    def _resolve_persistence_path(self) -> Path | None:
        manager = self.session_manager_provider()
        persistence = (
            getattr(manager, "persistence", None) if manager is not None else None
        )
        storage_path = getattr(persistence, "storage_path", None)
        if storage_path is not None:
            return Path(storage_path)
        return self.persistence_path

    def _overall_status(self, components: list[ComponentHealth]) -> HealthStatus:
        statuses = [component.status for component in components]
        if any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        if any(status == HealthStatus.DEGRADED for status in statuses):
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY


def _probe_storage_path(storage_path: Path) -> None:
    storage_path.mkdir(parents=True, exist_ok=True)
    probe_path = storage_path / ".health_check"
    probe_path.write_text("ok", encoding="utf-8")
    probe_path.unlink(missing_ok=True)


def _session_status(session: object) -> str:
    status = getattr(session, "status", "unknown")
    return str(getattr(status, "value", status)).lower()


def _session_error_count(session: object) -> int:
    state = getattr(session, "state", None)
    value = getattr(state, "error_count", 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


__all__ = ["ComponentHealth", "HealthChecker", "HealthStatus"]
