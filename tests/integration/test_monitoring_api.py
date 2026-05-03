"""Integration tests for monitoring API endpoints."""

from __future__ import annotations

from algotrading.api.config import APIConfig
from algotrading.api.main import create_app
from algotrading.src.monitoring import MetricsCollector
from fastapi.testclient import TestClient


def _client() -> TestClient:
    config = APIConfig(api_key="monitoring-secret-key", debug=True)
    app = create_app(config)
    return TestClient(app)


def test_health_endpoint_returns_components_and_correlation_header() -> None:
    """Health checks should expose component detail and preserve correlation IDs."""

    MetricsCollector().reset()
    client = _client()

    response = client.get("/health", headers={"X-Correlation-ID": "trace-health"})

    assert response.status_code == 200
    assert response.headers["X-Correlation-ID"] == "trace-health"
    payload = response.json()
    assert payload["status"] == "healthy"
    assert {component["name"] for component in payload["components"]} == {
        "broker",
        "persistence",
        "sessions",
    }


def test_metrics_endpoints_are_prometheus_and_json_compatible() -> None:
    """Metrics endpoints should expose request metrics without authentication."""

    collector = MetricsCollector()
    collector.reset()
    client = _client()
    health_response = client.get("/health")
    assert health_response.status_code == 200

    prometheus_response = client.get("/metrics")
    assert prometheus_response.status_code == 200
    assert "text/plain" in prometheus_response.headers["content-type"]
    assert "http_requests_total" in prometheus_response.text

    json_response = client.get("/metrics/json")
    assert json_response.status_code == 200
    assert (
        "http_requests_total{method=GET,path=/health,status=200}"
        in json_response.json()["counters"]
    )
