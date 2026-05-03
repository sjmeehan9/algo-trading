"""Unit tests for metrics collection and Prometheus export."""

from __future__ import annotations

from algotrading.src.monitoring import MetricsCollector


def test_metrics_collector_records_counters_gauges_and_observations() -> None:
    """MetricsCollector should aggregate values by metric name and tags."""

    collector = MetricsCollector()
    collector.reset()

    collector.increment("orders_total", tags={"session": "s1", "action": "BUY"})
    collector.increment("orders_total", tags={"session": "s1", "action": "BUY"})
    collector.gauge("position_quantity", 3.0, tags={"session": "s1", "symbol": "AAPL"})
    collector.record("decision_latency_ms", 10.0, tags={"session": "s1"})
    collector.record("decision_latency_ms", 20.0, tags={"session": "s1"})

    payload = collector.get_all()

    assert payload["counters"]["orders_total{action=BUY,session=s1}"] == 2
    assert payload["gauges"]["position_quantity{session=s1,symbol=AAPL}"] == 3.0
    assert payload["metrics"]["decision_latency_ms{session=s1}"]["count"] == 2
    assert payload["metrics"]["decision_latency_ms{session=s1}"]["mean"] == 15.0


def test_metrics_collector_exports_prometheus_text() -> None:
    """Prometheus export should contain valid metric lines with labels."""

    collector = MetricsCollector()
    collector.reset()
    collector.increment("http_requests_total", tags={"method": "GET", "status": 200})
    collector.record("http_request_latency_ms", 7.5, tags={"method": "GET"})

    text = collector.get_prometheus_format()

    assert "# TYPE http_requests_total counter" in text
    assert 'http_requests_total{method="GET",status="200"} 1' in text
    assert "# TYPE http_request_latency_ms_count counter" in text
    assert 'http_request_latency_ms_count{method="GET"} 1' in text
