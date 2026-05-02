"""Trading decision container for live inference output."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from math import isfinite

_VALID_ACTIONS = {"BUY", "SELL", "HOLD"}


@dataclass(slots=True)
class TradingDecision:
    """Decision emitted by the real-time inference pipeline.

    Args:
        timestamp: Market timestamp associated with the decision.
        action: Trading action, one of ``BUY``, ``SELL``, or ``HOLD``.
        confidence: Confidence score in the inclusive range ``[0, 1]``.
        market_price: Latest market price used by the decision.
        signals_used: Supporting signal identifiers used with fresh values.
        latency_ms: Total decision latency in milliseconds.
        metadata: Additional diagnostic context for monitoring and audit logs.
    """

    timestamp: datetime
    action: str
    confidence: float
    market_price: float
    signals_used: list[str]
    latency_ms: float
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate decision fields."""

        normalized_action = self.action.upper()
        if normalized_action not in _VALID_ACTIONS:
            raise ValueError("action must be one of BUY, SELL, or HOLD")
        self.action = normalized_action

        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and between 0 and 1")
        if not isfinite(self.market_price) or self.market_price < 0:
            raise ValueError("market_price must be finite and non-negative")
        if not isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("latency_ms must be finite and non-negative")

    def should_execute(self, min_confidence: float = 0.5) -> bool:
        """Return whether this decision should place a live order.

        Args:
            min_confidence: Required confidence threshold for order execution.

        Returns:
            ``True`` when the action is not ``HOLD`` and confidence is high enough.
        """

        if not isfinite(min_confidence) or not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be finite and between 0 and 1")
        return self.action != "HOLD" and self.confidence >= min_confidence

    def to_dict(self) -> dict[str, object]:
        """Serialize the decision for API responses, logs, or persistence."""

        return {
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "confidence": self.confidence,
            "market_price": self.market_price,
            "signals_used": list(self.signals_used),
            "latency_ms": self.latency_ms,
            "metadata": dict(self.metadata),
        }
