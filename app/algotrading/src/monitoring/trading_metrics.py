"""Trading-specific metric recording helpers."""

from __future__ import annotations

from math import isfinite

from algotrading.src.monitoring.metrics import MetricsCollector


class TradingMetrics:
    """Record trading-session metrics into a shared metrics collector."""

    def __init__(self, metrics_collector: MetricsCollector | None = None) -> None:
        """Initialize trading metrics with a collector dependency."""

        self.metrics = metrics_collector or MetricsCollector()

    def record_order(
        self,
        session_id: str,
        action: str,
        quantity: float,
        price: float,
    ) -> None:
        """Record an order submission and its notional value."""

        tags = {
            "session": _normalize_session(session_id),
            "action": _normalize_action(action),
        }
        safe_quantity = _finite_or_zero(quantity)
        safe_price = _finite_or_zero(price)
        self.metrics.increment("orders_total", tags=tags)
        self.metrics.record("order_value", safe_quantity * safe_price, tags=tags)
        self.metrics.gauge("last_order_quantity", safe_quantity, tags=tags)

    def record_fill(
        self,
        session_id: str,
        action: str,
        quantity: float,
        price: float,
        slippage: float,
    ) -> None:
        """Record a broker fill and execution slippage."""

        tags = {
            "session": _normalize_session(session_id),
            "action": _normalize_action(action),
        }
        self.metrics.increment("fills_total", tags=tags)
        self.metrics.record(
            "fill_value", _finite_or_zero(quantity) * _finite_or_zero(price), tags=tags
        )
        self.metrics.record("fill_slippage", abs(_finite_or_zero(slippage)), tags=tags)

    def record_order_status(
        self,
        session_id: str,
        status: str,
    ) -> None:
        """Record a broker order status callback."""

        normalized_status = status.strip().lower() or "unknown"
        self.metrics.increment(
            "order_status_total",
            tags={
                "session": _normalize_session(session_id),
                "status": normalized_status,
            },
        )

    def record_decision(
        self,
        session_id: str,
        action: str,
        confidence: float,
        latency_ms: float,
    ) -> None:
        """Record a trading decision emitted by the inference pipeline."""

        tags = {
            "session": _normalize_session(session_id),
            "action": _normalize_action(action),
        }
        self.metrics.increment("decisions_total", tags=tags)
        self.metrics.record(
            "decision_confidence", _bounded_confidence(confidence), tags=tags
        )
        self.metrics.record(
            "decision_latency_ms", max(0.0, _finite_or_zero(latency_ms)), tags=tags
        )

    def update_pnl(
        self,
        session_id: str,
        realized_pnl: float,
        unrealized_pnl: float,
    ) -> None:
        """Update realized, unrealized, and total P&L gauges."""

        tags = {"session": _normalize_session(session_id)}
        realized = _finite_or_zero(realized_pnl)
        unrealized = _finite_or_zero(unrealized_pnl)
        self.metrics.gauge("realized_pnl", realized, tags=tags)
        self.metrics.gauge("unrealized_pnl", unrealized, tags=tags)
        self.metrics.gauge("total_pnl", realized + unrealized, tags=tags)

    def update_position(
        self,
        session_id: str,
        symbol: str,
        quantity: float,
        value: float,
    ) -> None:
        """Update current position quantity and market value gauges."""

        tags = {
            "session": _normalize_session(session_id),
            "symbol": symbol.strip().upper() or "UNKNOWN",
        }
        self.metrics.gauge("position_quantity", _finite_or_zero(quantity), tags=tags)
        self.metrics.gauge("position_value", _finite_or_zero(value), tags=tags)


def _normalize_session(session_id: str) -> str:
    normalized = session_id.strip()
    if not normalized:
        raise ValueError("session_id must be non-empty")
    return normalized


def _normalize_action(action: str) -> str:
    normalized = action.strip().upper()
    return normalized or "UNKNOWN"


def _finite_or_zero(value: float) -> float:
    numeric_value = float(value)
    return numeric_value if isfinite(numeric_value) else 0.0


def _bounded_confidence(value: float) -> float:
    numeric_value = _finite_or_zero(value)
    return max(0.0, min(1.0, numeric_value))


__all__ = ["TradingMetrics"]
