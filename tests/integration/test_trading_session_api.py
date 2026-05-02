"""Integration tests for trading session API endpoints."""

from __future__ import annotations

from datetime import UTC, datetime

from algotrading.api.config import APIConfig
from algotrading.api.main import create_app
from fastapi.testclient import TestClient


class _FakeSession:
    """Minimal session object exposed by the fake API manager."""

    def __init__(self, session_id: str, status: str = "created") -> None:
        self.session_id = session_id
        self.status = status

    def get_status(self) -> dict[str, object]:
        """Return an API-compatible status payload."""

        return {
            "session_id": self.session_id,
            "status": self.status,
            "model_id": "core-1",
            "generation_id": "gen-1",
            "broker": "mock",
            "mode": "paper",
            "symbols": ["AAPL"],
            "supporting_model_ids": [],
            "created_at": datetime.now(tz=UTC).isoformat(),
            "started_at": None,
            "stopped_at": None,
            "positions": {},
            "orders_count": 0,
            "decisions_count": 0,
            "executed_orders_count": 0,
            "error_count": 0,
            "error_message": None,
            "last_price": None,
            "last_decision": None,
        }


class _FakeSessionManager:
    """API-facing fake manager with the TradingSessionManager method shape."""

    def __init__(self) -> None:
        self.sessions: dict[str, _FakeSession] = {}
        self.initialized = False
        self.shutdown_called = False

    async def initialize(self) -> None:
        """Mark the manager initialized."""

        self.initialized = True

    async def create_session(self, **kwargs) -> _FakeSession:
        """Create a fake session and record the request payload."""

        del kwargs
        session = _FakeSession("session-1")
        self.sessions[session.session_id] = session
        return session

    async def list_sessions(self, status=None) -> list[_FakeSession]:
        """List fake sessions, optionally filtered by status."""

        sessions = list(self.sessions.values())
        if status is not None:
            sessions = [
                session for session in sessions if session.status == status.value
            ]
        return sessions

    async def get_session(self, session_id: str) -> _FakeSession:
        """Return one fake session."""

        return self.sessions[session_id]

    async def start_session(self, session_id: str) -> _FakeSession:
        """Mark one fake session running."""

        session = self.sessions[session_id]
        session.status = "running"
        return session

    async def pause_session(self, session_id: str) -> _FakeSession:
        """Mark one fake session paused."""

        session = self.sessions[session_id]
        session.status = "paused"
        return session

    async def stop_session(
        self,
        session_id: str,
        close_positions: bool = False,
    ) -> _FakeSession:
        """Mark one fake session stopped."""

        del close_positions
        session = self.sessions[session_id]
        session.status = "stopped"
        return session

    async def shutdown(self, close_positions: bool = False) -> None:
        """Record application shutdown calls."""

        del close_positions
        self.shutdown_called = True


def _client() -> tuple[TestClient, _FakeSessionManager]:
    """Build an API client with a fake trading session manager."""

    app = create_app(APIConfig(api_key="trading-secret-key", debug=True))
    manager = _FakeSessionManager()
    app.state.trading_session_manager = manager
    return TestClient(app), manager


def _headers() -> dict[str, str]:
    """Return API authentication headers."""

    return {"X-API-Key": "trading-secret-key"}


def test_trading_session_api_lifecycle() -> None:
    """Trading session endpoints should create and transition sessions."""

    client, manager = _client()

    create_response = client.post(
        "/api/v1/trading/sessions",
        headers=_headers(),
        json={
            "model_id": "core-1",
            "generation_id": "gen-1",
            "mode": "paper",
            "broker": "mock",
            "symbols": ["aapl"],
        },
    )
    assert create_response.status_code == 201, create_response.text
    assert create_response.json()["data"]["session_id"] == "session-1"
    assert manager.initialized is True

    list_response = client.get("/api/v1/trading/sessions", headers=_headers())
    assert list_response.status_code == 200
    assert len(list_response.json()["data"]) == 1

    start_response = client.post(
        "/api/v1/trading/sessions/session-1/start",
        headers=_headers(),
    )
    assert start_response.status_code == 200
    assert start_response.json()["data"]["status"] == "running"

    pause_response = client.post(
        "/api/v1/trading/sessions/session-1/pause",
        headers=_headers(),
    )
    assert pause_response.status_code == 200
    assert pause_response.json()["data"]["status"] == "paused"

    stop_response = client.post(
        "/api/v1/trading/sessions/session-1/stop",
        headers=_headers(),
        json={"close_positions": True},
    )
    assert stop_response.status_code == 200
    assert stop_response.json()["data"]["status"] == "stopped"


def test_trading_session_api_filters_by_status() -> None:
    """Session listing should accept a lifecycle status filter."""

    client, manager = _client()
    manager.sessions["session-1"] = _FakeSession("session-1", status="running")
    manager.sessions["session-2"] = _FakeSession("session-2", status="paused")

    response = client.get(
        "/api/v1/trading/sessions?status=paused",
        headers=_headers(),
    )

    assert response.status_code == 200, response.text
    sessions = response.json()["data"]
    assert [session["session_id"] for session in sessions] == ["session-2"]
