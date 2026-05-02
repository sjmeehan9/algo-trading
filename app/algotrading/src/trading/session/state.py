"""Durable state containers for live trading sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import isfinite

from algotrading.src.broker import PositionInfo
from algotrading.src.trading.inference import TradingDecision


@dataclass(slots=True)
class SessionOrder:
    """Order metadata recorded by a trading session.

    Args:
        order_id: Broker-assigned order identifier.
        symbol: Traded instrument symbol.
        action: Trading action, ``BUY`` or ``SELL``.
        quantity: Submitted order quantity.
        price: Decision market price when the order was submitted.
        confidence: Decision confidence at order time.
        decision_timestamp: Timestamp of the decision that triggered the order.
        status: Latest order lifecycle status known to the session.
        created_at: Timestamp when the session recorded the order.
    """

    order_id: str
    symbol: str
    action: str
    quantity: float
    price: float
    confidence: float
    decision_timestamp: datetime
    status: str = "submitted"
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))

    def __post_init__(self) -> None:
        """Validate and normalize order fields."""

        self.order_id = self.order_id.strip()
        self.symbol = self.symbol.strip().upper()
        self.action = self.action.strip().upper()
        self.status = self.status.strip().lower()

        if not self.order_id:
            raise ValueError("order_id must be non-empty")
        if not self.symbol:
            raise ValueError("symbol must be non-empty")
        if self.action not in {"BUY", "SELL"}:
            raise ValueError("action must be BUY or SELL")
        if self.quantity <= 0 or not isfinite(self.quantity):
            raise ValueError("quantity must be finite and greater than zero")
        if self.price < 0 or not isfinite(self.price):
            raise ValueError("price must be finite and non-negative")
        if not 0.0 <= self.confidence <= 1.0 or not isfinite(self.confidence):
            raise ValueError("confidence must be finite and between 0 and 1")
        self.decision_timestamp = _ensure_aware(self.decision_timestamp)
        self.created_at = _ensure_aware(self.created_at)

    def to_dict(self) -> dict[str, object]:
        """Serialize order metadata to JSON-compatible values."""

        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "action": self.action,
            "quantity": self.quantity,
            "price": self.price,
            "confidence": self.confidence,
            "decision_timestamp": self.decision_timestamp.isoformat(),
            "status": self.status,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SessionOrder":
        """Deserialize order metadata from JSON-compatible values."""

        return cls(
            order_id=str(payload["order_id"]),
            symbol=str(payload["symbol"]),
            action=str(payload["action"]),
            quantity=float(payload["quantity"]),
            price=float(payload["price"]),
            confidence=float(payload["confidence"]),
            decision_timestamp=_parse_datetime(str(payload["decision_timestamp"])),
            status=str(payload.get("status") or "submitted"),
            created_at=_parse_datetime(str(payload["created_at"])),
        )


@dataclass(slots=True)
class SessionState:
    """Mutable state tracked for one live trading session."""

    session_id: str
    positions: dict[str, float] = field(default_factory=dict)
    orders: dict[str, SessionOrder] = field(default_factory=dict)
    last_price: float | None = None
    last_decision: TradingDecision | None = None
    decisions_count: int = 0
    executed_orders_count: int = 0
    error_count: int = 0
    error_message: str | None = None
    updated_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))

    def __post_init__(self) -> None:
        """Validate state identifiers and counters."""

        self.session_id = self.session_id.strip()
        if not self.session_id:
            raise ValueError("session_id must be non-empty")
        if self.decisions_count < 0:
            raise ValueError("decisions_count must be non-negative")
        if self.executed_orders_count < 0:
            raise ValueError("executed_orders_count must be non-negative")
        if self.error_count < 0:
            raise ValueError("error_count must be non-negative")
        self.positions = {
            symbol.strip().upper(): float(quantity)
            for symbol, quantity in self.positions.items()
            if symbol.strip()
        }
        self.updated_at = _ensure_aware(self.updated_at)

    def record_decision(self, decision: TradingDecision) -> None:
        """Record the latest generated trading decision."""

        self.last_decision = decision
        self.last_price = decision.market_price
        self.decisions_count += 1
        self.touch()

    def record_order(
        self,
        *,
        order_id: str,
        symbol: str,
        action: str,
        quantity: float,
        decision: TradingDecision,
        status: str = "submitted",
    ) -> SessionOrder:
        """Record a broker order and update optimistic position state."""

        order = SessionOrder(
            order_id=order_id,
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=decision.market_price,
            confidence=decision.confidence,
            decision_timestamp=decision.timestamp,
            status=status,
        )
        self.orders[order.order_id] = order
        self.executed_orders_count += 1
        self.adjust_position(
            order.symbol, quantity if order.action == "BUY" else -quantity
        )
        self.touch()
        return order

    def update_order_status(self, order_id: str, status: str) -> None:
        """Update the lifecycle status for a previously recorded order."""

        order = self.orders.get(order_id)
        if order is None:
            return
        order.status = status.strip().lower()
        self.touch()

    def record_error(self, message: str) -> None:
        """Record a session error without raising it further."""

        self.error_count += 1
        self.error_message = message
        self.touch()

    def set_position(self, symbol: str, quantity: float) -> None:
        """Set an absolute position quantity for a symbol."""

        normalized_symbol = _normalize_symbol(symbol)
        if abs(quantity) < 1e-9:
            self.positions.pop(normalized_symbol, None)
        else:
            self.positions[normalized_symbol] = float(quantity)
        self.touch()

    def adjust_position(self, symbol: str, quantity_delta: float) -> None:
        """Adjust a position quantity by a delta."""

        normalized_symbol = _normalize_symbol(symbol)
        current = self.positions.get(normalized_symbol, 0.0)
        self.set_position(normalized_symbol, current + float(quantity_delta))

    def get_position(self, symbol: str) -> float:
        """Return the tracked position quantity for one symbol."""

        return self.positions.get(_normalize_symbol(symbol), 0.0)

    def sync_positions(self, positions: list[PositionInfo]) -> None:
        """Replace tracked positions with a broker position snapshot."""

        self.positions.clear()
        for position in positions:
            self.set_position(position.contract.symbol, float(position.quantity))
        self.touch()

    def touch(self) -> None:
        """Update the state modification timestamp."""

        self.updated_at = datetime.now(tz=UTC)

    def to_dict(self) -> dict[str, object]:
        """Serialize state to JSON-compatible values."""

        return {
            "session_id": self.session_id,
            "positions": dict(self.positions),
            "orders": {
                order_id: order.to_dict() for order_id, order in self.orders.items()
            },
            "last_price": self.last_price,
            "last_decision": (
                self.last_decision.to_dict() if self.last_decision is not None else None
            ),
            "decisions_count": self.decisions_count,
            "executed_orders_count": self.executed_orders_count,
            "error_count": self.error_count,
            "error_message": self.error_message,
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SessionState":
        """Deserialize state from JSON-compatible values."""

        raw_orders = dict(payload.get("orders") or {})
        raw_decision = payload.get("last_decision")
        return cls(
            session_id=str(payload["session_id"]),
            positions={
                str(symbol): float(quantity)
                for symbol, quantity in dict(payload.get("positions") or {}).items()
            },
            orders={
                str(order_id): SessionOrder.from_dict(dict(order_payload))
                for order_id, order_payload in raw_orders.items()
            },
            last_price=(
                float(payload["last_price"])
                if payload.get("last_price") is not None
                else None
            ),
            last_decision=(
                _decision_from_dict(dict(raw_decision))
                if isinstance(raw_decision, dict)
                else None
            ),
            decisions_count=int(payload.get("decisions_count") or 0),
            executed_orders_count=int(payload.get("executed_orders_count") or 0),
            error_count=int(payload.get("error_count") or 0),
            error_message=(
                str(payload["error_message"])
                if payload.get("error_message") is not None
                else None
            ),
            updated_at=_parse_datetime(str(payload["updated_at"])),
        )


def _decision_from_dict(payload: dict[str, object]) -> TradingDecision:
    return TradingDecision(
        timestamp=_parse_datetime(str(payload["timestamp"])),
        action=str(payload["action"]),
        confidence=float(payload["confidence"]),
        market_price=float(payload["market_price"]),
        signals_used=[str(item) for item in list(payload.get("signals_used") or [])],
        latency_ms=float(payload["latency_ms"]),
        metadata=dict(payload.get("metadata") or {}),
    )


def _normalize_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if not normalized:
        raise ValueError("symbol must be non-empty")
    return normalized


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return _ensure_aware(parsed)


def _ensure_aware(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp


__all__ = ["SessionOrder", "SessionState"]
