"""Unit tests for trading decision semantics."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from algotrading.src.trading.inference import TradingDecision


def test_trading_decision_execute_thresholds_and_serialization() -> None:
    """Executable decisions require non-hold actions and enough confidence."""

    timestamp = datetime(2026, 1, 2, 14, 30, tzinfo=UTC)
    decision = TradingDecision(
        timestamp=timestamp,
        action="buy",
        confidence=0.75,
        market_price=101.25,
        signals_used=["sentiment"],
        latency_ms=12.0,
        metadata={"source": "test"},
    )

    assert decision.action == "BUY"
    assert decision.should_execute(min_confidence=0.7) is True
    assert decision.should_execute(min_confidence=0.8) is False

    payload = decision.to_dict()
    assert payload["timestamp"] == timestamp.isoformat()
    assert payload["action"] == "BUY"
    assert payload["signals_used"] == ["sentiment"]


def test_hold_decision_never_executes() -> None:
    """Hold decisions should not place orders even with high confidence."""

    decision = TradingDecision(
        timestamp=datetime(2026, 1, 2, 14, 30, tzinfo=UTC),
        action="HOLD",
        confidence=1.0,
        market_price=101.25,
        signals_used=[],
        latency_ms=1.0,
    )

    assert decision.should_execute(min_confidence=0.1) is False


def test_trading_decision_rejects_invalid_values() -> None:
    """Invalid action and confidence values should fail fast."""

    timestamp = datetime(2026, 1, 2, 14, 30, tzinfo=UTC)
    with pytest.raises(ValueError, match="action"):
        TradingDecision(
            timestamp=timestamp,
            action="WAIT",
            confidence=0.5,
            market_price=100.0,
            signals_used=[],
            latency_ms=1.0,
        )

    with pytest.raises(ValueError, match="confidence"):
        TradingDecision(
            timestamp=timestamp,
            action="BUY",
            confidence=1.5,
            market_price=100.0,
            signals_used=[],
            latency_ms=1.0,
        )
