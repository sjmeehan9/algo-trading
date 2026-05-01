"""Unit tests for API authentication middleware."""

from __future__ import annotations

from algotrading.api.config import APIConfig
from algotrading.api.main import create_app
from fastapi import Request
from fastapi.testclient import TestClient


def _make_client() -> TestClient:
    config = APIConfig(
        api_key="unit-secret-key",
        debug=True,
        cors_origins=["http://localhost:3000"],
    )
    app = create_app(config)

    @app.get("/protected")
    async def protected(request: Request) -> dict[str, bool]:
        return {"authenticated": bool(getattr(request.state, "authenticated", False))}

    return TestClient(app)


def test_auth_accepts_valid_api_key_header() -> None:
    """Protected routes should succeed with the configured API key header."""

    client = _make_client()

    response = client.get("/protected", headers={"X-API-Key": "unit-secret-key"})

    assert response.status_code == 200
    assert response.json() == {"authenticated": True}


def test_auth_accepts_valid_api_key_query_param() -> None:
    """Protected routes should also accept API key via query parameters."""

    client = _make_client()

    response = client.get("/protected?api_key=unit-secret-key")

    assert response.status_code == 200
    assert response.json() == {"authenticated": True}


def test_auth_rejects_invalid_api_key() -> None:
    """Protected routes should return 401 for invalid API keys."""

    client = _make_client()

    response = client.get("/protected", headers={"X-API-Key": "wrong-key"})

    assert response.status_code == 401
    body = response.json()
    assert body["success"] is False
    assert body["error_code"] == "AUTH_INVALID_API_KEY"


def test_auth_rejects_missing_api_key() -> None:
    """Protected routes should return 401 when no API key is supplied."""

    client = _make_client()

    response = client.get("/protected")

    assert response.status_code == 401
    body = response.json()
    assert body["success"] is False
    assert body["error_code"] == "AUTH_MISSING_API_KEY"


def test_auth_exempts_health_endpoint() -> None:
    """Health check endpoint should remain accessible without auth."""

    client = _make_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
