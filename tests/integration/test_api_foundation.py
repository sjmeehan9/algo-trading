"""Integration tests for Phase 5.1 API foundation."""

from __future__ import annotations

from algotrading.api.config import APIConfig
from algotrading.api.main import create_app
from fastapi.testclient import TestClient


def _make_client() -> TestClient:
    config = APIConfig(
        api_key="integration-secret-key",
        debug=True,
        cors_origins=["http://localhost:3000"],
    )
    app = create_app(config)
    return TestClient(app)


def test_health_check_returns_healthy_status() -> None:
    """Application startup should expose a healthy health endpoint."""

    client = _make_client()

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert "timestamp" in payload


def test_cors_header_is_present_for_allowed_origin() -> None:
    """CORS middleware should return allow-origin header for allowed origins."""

    client = _make_client()

    response = client.get(
        "/health",
        headers={"Origin": "http://localhost:3000"},
    )

    assert response.status_code == 200
    assert (
        response.headers.get("access-control-allow-origin") == "http://localhost:3000"
    )


def test_docs_endpoint_available_when_debug_enabled() -> None:
    """Swagger docs should be reachable for local development workflows."""

    client = _make_client()

    response = client.get("/docs")

    assert response.status_code == 200
    assert "swagger" in response.text.lower()


def test_websocket_connect_and_message_exchange() -> None:
    """WebSocket endpoint should accept authenticated clients and respond to commands."""

    client = _make_client()

    with client.websocket_connect(
        "/ws?client_id=ws-client-1&api_key=integration-secret-key"
    ) as websocket:
        connected = websocket.receive_json()
        assert connected["type"] == "connected"
        assert connected["client_id"] == "ws-client-1"

        websocket.send_json({"type": "subscribe", "topic": "training:model-1"})
        subscribed = websocket.receive_json()
        assert subscribed == {"type": "subscribed", "topic": "training:model-1"}

        websocket.send_json({"type": "ping"})
        pong = websocket.receive_json()
        assert pong == {"type": "pong"}
