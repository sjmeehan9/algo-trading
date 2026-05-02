"""Unit tests for trading session state containers."""

from __future__ import annotations

from datetime import UTC, datetime

from algotrading.src.trading.inference import TradingDecision
from algotrading.src.trading.session import SessionState


def _decision(action: str = "BUY") -> TradingDecision:
    """Build a deterministic trading decision for state tests."""

    return TradingDecision(
        timestamp=datetime(2026, 5, 3, 14, 30, tzinfo=UTC),
        action=action,
        confidence=0.9,
        market_price=100.0,
        signals_used=["sentiment"],
        latency_ms=2.5,
    )


def test_session_state_records_decisions_orders_and_positions() -> None:
    """Session state should track decisions, orders, and position deltas."""

    state = SessionState(session_id="session-1")
    decision = _decision("BUY")

    state.record_decision(decision)
    order = state.record_order(
        order_id="order-1",
        symbol="aapl",
        action="BUY",
        quantity=3.0,
        decision=decision,
    )

    assert state.decisions_count == 1
    assert state.executed_orders_count == 1
    assert state.last_price == 100.0
    assert order.symbol == "AAPL"
    assert state.get_position("AAPL") == 3.0

    state.update_order_status("order-1", "FILLED")
    assert state.orders["order-1"].status == "filled"


def test_session_state_round_trips_to_dict() -> None:
    """Persisted state payloads should reconstruct equivalent state."""

    state = SessionState(session_id="session-1")
    state.record_decision(_decision("SELL"))
    state.record_order(
        order_id="order-1",
        symbol="MSFT",
        action="SELL",
        quantity=2.0,
        decision=_decision("SELL"),
    )
    state.record_error("temporary failure")

    restored = SessionState.from_dict(state.to_dict())

    assert restored.session_id == state.session_id
    assert restored.get_position("MSFT") == -2.0
    assert restored.orders["order-1"].action == "SELL"
    assert restored.last_decision is not None
    assert restored.last_decision.action == "SELL"
    assert restored.error_count == 1
    assert restored.error_message == "temporary failure"
