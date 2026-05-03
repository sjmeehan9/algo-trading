"""Unit tests for trading-specific metrics."""

from __future__ import annotations

from algotrading.src.monitoring import MetricsCollector, TradingMetrics


def test_trading_metrics_record_decisions_orders_and_positions() -> None:
    """TradingMetrics should populate decision, order, P&L, and position series."""

    collector = MetricsCollector()
    collector.reset()
    trading_metrics = TradingMetrics(collector)

    trading_metrics.record_decision("session-1", "BUY", 0.75, 12.5)
    trading_metrics.record_order("session-1", "BUY", 3.0, 100.0)
    trading_metrics.record_fill("session-1", "BUY", 3.0, 100.1, 0.1)
    trading_metrics.update_position("session-1", "AAPL", 3.0, 300.0)
    trading_metrics.update_pnl("session-1", 5.0, -1.0)

    payload = collector.get_all()

    assert payload["counters"]["decisions_total{action=BUY,session=session-1}"] == 1
    assert payload["counters"]["orders_total{action=BUY,session=session-1}"] == 1
    assert payload["counters"]["fills_total{action=BUY,session=session-1}"] == 1
    assert payload["gauges"]["position_quantity{session=session-1,symbol=AAPL}"] == 3.0
    assert payload["gauges"]["total_pnl{session=session-1}"] == 4.0
