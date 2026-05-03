"""Unit tests for monitoring health checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from algotrading.src.broker import BrokerRegistry
from algotrading.src.monitoring import HealthChecker


@dataclass
class _FakeState:
    """Minimal session state test double."""

    error_count: int = 0


@dataclass
class _FakeSession:
    """Minimal session test double for health checks."""

    status: str
    state: _FakeState


class _FakePersistence:
    """Persistence test double exposing a storage path."""

    def __init__(self, storage_path: Path) -> None:
        self.storage_path = storage_path


class _FakeManager:
    """Session manager test double with in-memory sessions."""

    def __init__(self, storage_path: Path) -> None:
        self.persistence = _FakePersistence(storage_path)
        self._sessions: dict[str, _FakeSession] = {}


@pytest.mark.asyncio
async def test_health_checker_reports_healthy_without_active_runtime() -> None:
    """A freshly started API should be healthy before broker/session startup."""

    checker = HealthChecker(
        broker_registry=BrokerRegistry.isolated(include_builtins=False)
    )

    result = await checker.check_all()

    assert result["status"] == "healthy"
    assert {component["name"] for component in result["components"]} == {
        "broker",
        "persistence",
        "sessions",
    }


@pytest.mark.asyncio
async def test_health_checker_marks_session_errors_as_degraded(tmp_path: Path) -> None:
    """Errored sessions should degrade health without failing the whole service."""

    manager = _FakeManager(tmp_path / "sessions")
    manager._sessions["session-1"] = _FakeSession(
        status="error",
        state=_FakeState(error_count=2),
    )
    checker = HealthChecker(session_manager_provider=lambda: manager)

    result = await checker.check_all()

    sessions_component = next(
        component
        for component in result["components"]
        if component["name"] == "sessions"
    )

    assert result["status"] == "degraded"
    assert sessions_component["metadata"]["errored"] == 1
    assert sessions_component["metadata"]["error_count"] == 2
