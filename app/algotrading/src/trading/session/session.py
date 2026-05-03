"""Lifecycle management for a single live trading session."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from math import floor, isfinite
from typing import Protocol

from algotrading.src.broker import (
    BarData,
    BrokerAdapter,
    ContractSpec,
    InstrumentType,
    OrderSide,
    OrderSpec,
    OrderStatus,
    OrderType,
    PositionInfo,
)
from algotrading.src.monitoring import TradingMetrics
from algotrading.src.trading.inference import TradingDecision
from algotrading.src.trading.session.exceptions import InvalidSessionStateError
from algotrading.src.trading.session.persistence import SessionPersistence
from algotrading.src.trading.session.state import SessionState

logger = logging.getLogger(__name__)


class SessionStatus(str, Enum):
    """Lifecycle states for live trading sessions."""

    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class InferencePipelineProtocol(Protocol):
    """Runtime protocol implemented by real-time inference pipelines."""

    async def start(
        self,
        decision_callback: Callable[[TradingDecision], None] | None = None,
    ) -> None:
        """Start accepting market data."""

    async def stop(self) -> None:
        """Stop accepting market data."""

    async def process_market_data(self, bar_data: BarData) -> TradingDecision:
        """Convert a broker market bar into a trading decision."""


@dataclass(slots=True)
class SessionConfig:
    """Configuration for one trading session."""

    model_id: str
    generation_id: str
    broker_name: str
    mode: str
    symbols: list[str]
    supporting_model_ids: list[str] = field(default_factory=list)
    risk_config: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize session configuration."""

        self.model_id = self.model_id.strip()
        self.generation_id = self.generation_id.strip()
        self.broker_name = self.broker_name.strip().lower()
        self.mode = self.mode.strip().lower()
        self.symbols = [
            symbol.strip().upper() for symbol in self.symbols if symbol.strip()
        ]
        self.supporting_model_ids = [
            model_id.strip()
            for model_id in self.supporting_model_ids
            if model_id.strip()
        ]

        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        if not self.generation_id:
            raise ValueError("generation_id must be non-empty")
        if not self.broker_name:
            raise ValueError("broker_name must be non-empty")
        if self.mode not in {"paper", "live"}:
            raise ValueError("mode must be paper or live")
        if not self.symbols:
            raise ValueError("at least one trading symbol is required")

    def to_dict(self) -> dict[str, object]:
        """Serialize the session configuration."""

        return {
            "model_id": self.model_id,
            "generation_id": self.generation_id,
            "broker_name": self.broker_name,
            "mode": self.mode,
            "symbols": list(self.symbols),
            "supporting_model_ids": list(self.supporting_model_ids),
            "risk_config": dict(self.risk_config),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SessionConfig":
        """Deserialize a session configuration."""

        return cls(
            model_id=str(payload["model_id"]),
            generation_id=str(payload["generation_id"]),
            broker_name=str(payload["broker_name"]),
            mode=str(payload["mode"]),
            symbols=[str(symbol) for symbol in list(payload.get("symbols") or [])],
            supporting_model_ids=[
                str(model_id)
                for model_id in list(payload.get("supporting_model_ids") or [])
            ],
            risk_config=dict(payload.get("risk_config") or {}),
        )


class TradingSession:
    """Manage one deployed model trading on one broker adapter."""

    def __init__(
        self,
        session_id: str,
        config: SessionConfig,
        broker: BrokerAdapter,
        inference_pipeline: InferencePipelineProtocol,
        persistence: SessionPersistence,
        *,
        state: SessionState | None = None,
        status: SessionStatus = SessionStatus.CREATED,
        created_at: datetime | None = None,
        started_at: datetime | None = None,
        stopped_at: datetime | None = None,
        trading_metrics: TradingMetrics | None = None,
    ) -> None:
        """Initialize a trading session.

        Args:
            session_id: Unique session identifier.
            config: Deployment and risk configuration.
            broker: Connected broker adapter used for data and orders.
            inference_pipeline: Real-time decision pipeline.
            persistence: Session persistence layer.
            state: Optional recovered session state.
            status: Initial lifecycle status.
            created_at: Original creation timestamp for recovered sessions.
            started_at: Original start timestamp for recovered sessions.
            stopped_at: Original stop timestamp for recovered sessions.
            trading_metrics: Optional metrics recorder for session events.
        """

        self.session_id = session_id.strip()
        if not self.session_id:
            raise ValueError("session_id must be non-empty")

        self.config = config
        self.broker = broker
        self.inference_pipeline = inference_pipeline
        self.persistence = persistence
        self.status = status
        self.state = state or SessionState(session_id=self.session_id)
        self.trading_metrics = trading_metrics or TradingMetrics()
        self.created_at = _ensure_aware(created_at or datetime.now(tz=UTC))
        self.started_at = _ensure_optional_aware(started_at)
        self.stopped_at = _ensure_optional_aware(stopped_at)

        self._data_subscription_ids: list[int] = []
        self._running = status == SessionStatus.RUNNING
        self._loop: asyncio.AbstractEventLoop | None = None
        self._callbacks_registered = False
        self._persist_every_decisions = int(
            self.config.risk_config.get("persist_every_decisions", 100)
        )
        if self._persist_every_decisions <= 0:
            self._persist_every_decisions = 100

    @classmethod
    def from_record(
        cls,
        record: dict[str, object],
        *,
        broker: BrokerAdapter,
        inference_pipeline: InferencePipelineProtocol,
        persistence: SessionPersistence,
    ) -> "TradingSession":
        """Reconstruct a session object from persisted metadata."""

        status = SessionStatus(str(record.get("status") or SessionStatus.CREATED.value))
        if status in {
            SessionStatus.RUNNING,
            SessionStatus.STARTING,
            SessionStatus.STOPPING,
        }:
            status = SessionStatus.PAUSED

        return cls(
            session_id=str(record["session_id"]),
            config=SessionConfig.from_dict(dict(record["config"])),
            broker=broker,
            inference_pipeline=inference_pipeline,
            persistence=persistence,
            state=SessionState.from_dict(dict(record["state"])),
            status=status,
            created_at=_parse_datetime(str(record["created_at"])),
            started_at=_parse_optional_datetime(record.get("started_at")),
            stopped_at=_parse_optional_datetime(record.get("stopped_at")),
        )

    async def start(self) -> None:
        """Start or resume the trading session."""

        if self.status not in {SessionStatus.CREATED, SessionStatus.PAUSED}:
            raise InvalidSessionStateError(
                f"Cannot start session in state {self.status.value}"
            )

        self.status = SessionStatus.STARTING
        self._loop = asyncio.get_running_loop()
        logger.info("Starting trading session %s", self.session_id)

        try:
            self._register_broker_callbacks()
            self._hydrate_positions_from_broker()
            self._ensure_market_data_subscriptions()
            await self.inference_pipeline.start()
            self._running = True
            self.status = SessionStatus.RUNNING
            if self.started_at is None:
                self.started_at = datetime.now(tz=UTC)
            self.stopped_at = None
            await self._persist_state()
        except Exception as exc:
            self._running = False
            self.status = SessionStatus.ERROR
            self.state.record_error(str(exc))
            await self._persist_state()
            raise

        logger.info("Trading session %s started", self.session_id)

    async def pause(self) -> None:
        """Pause the session while preserving subscriptions and positions."""

        if self.status != SessionStatus.RUNNING:
            raise InvalidSessionStateError(
                f"Cannot pause session in state {self.status.value}"
            )

        self._running = False
        await self.inference_pipeline.stop()
        self.status = SessionStatus.PAUSED
        await self._persist_state()
        logger.info("Trading session %s paused", self.session_id)

    async def stop(self, close_positions: bool = False) -> None:
        """Stop the session and optionally flatten tracked positions."""

        if self.status in {SessionStatus.STOPPED, SessionStatus.STOPPING}:
            return

        self.status = SessionStatus.STOPPING
        self._running = False

        try:
            await self.inference_pipeline.stop()
            self._unsubscribe_market_data()
            if close_positions:
                await self._close_all_positions()
            self.status = SessionStatus.STOPPED
            self.stopped_at = datetime.now(tz=UTC)
        except Exception as exc:
            self.status = SessionStatus.ERROR
            self.state.record_error(str(exc))
            logger.exception("Error stopping trading session %s", self.session_id)
        finally:
            await self._persist_state()

        logger.info("Trading session %s stopped", self.session_id)

    async def process_market_data(self, bar_data: BarData) -> TradingDecision | None:
        """Process one market bar and execute an eligible decision."""

        if not self._running or self.status != SessionStatus.RUNNING:
            return None

        try:
            decision = await self.inference_pipeline.process_market_data(bar_data)
            self.state.record_decision(decision)
            self.trading_metrics.record_decision(
                self.session_id,
                decision.action,
                decision.confidence,
                decision.latency_ms,
            )
            if decision.should_execute(min_confidence=self._min_confidence()):
                await self._execute_decision(decision)
            if self.state.decisions_count % self._persist_every_decisions == 0:
                await self._persist_state()
            return decision
        except Exception as exc:
            self.state.record_error(str(exc))
            logger.exception(
                "Market data processing failed for session %s", self.session_id
            )
            await self._persist_state()
            return None

    def get_status(self) -> dict[str, object]:
        """Return a serializable status snapshot for API responses."""

        return {
            "session_id": self.session_id,
            "status": self.status.value,
            "model_id": self.config.model_id,
            "generation_id": self.config.generation_id,
            "broker": self.config.broker_name,
            "mode": self.config.mode,
            "symbols": list(self.config.symbols),
            "supporting_model_ids": list(self.config.supporting_model_ids),
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
            "positions": dict(self.state.positions),
            "orders_count": len(self.state.orders),
            "decisions_count": self.state.decisions_count,
            "executed_orders_count": self.state.executed_orders_count,
            "error_count": self.state.error_count,
            "error_message": self.state.error_message,
            "last_price": self.state.last_price,
            "last_decision": (
                self.state.last_decision.to_dict()
                if self.state.last_decision is not None
                else None
            ),
        }

    def to_record(self) -> dict[str, object]:
        """Serialize this session for durable persistence."""

        return {
            "session_id": self.session_id,
            "config": self.config.to_dict(),
            "status": self.status.value,
            "state": self.state.to_dict(),
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
        }

    def _register_broker_callbacks(self) -> None:
        if self._callbacks_registered:
            return
        self.broker.register_order_callback(self._on_order_update)
        self.broker.subscribe_position_updates(self._on_position_update)
        self._callbacks_registered = True

    def _hydrate_positions_from_broker(self) -> None:
        positions = self.broker.get_positions()
        self.state.sync_positions(positions)
        unrealized_pnl = 0.0
        for position in positions:
            symbol = position.contract.symbol
            market_value = float(position.market_value or 0.0)
            unrealized_pnl += float(position.unrealized_pnl or 0.0)
            self.trading_metrics.update_position(
                self.session_id,
                symbol,
                float(position.quantity),
                market_value,
            )
        self.trading_metrics.update_pnl(self.session_id, 0.0, unrealized_pnl)

    def _ensure_market_data_subscriptions(self) -> None:
        if self._data_subscription_ids:
            return
        bar_size = int(self.config.risk_config.get("bar_size_seconds", 5))
        data_type = str(self.config.risk_config.get("data_type", "TRADES"))
        for symbol in self.config.symbols:
            contract = self._symbol_to_contract(symbol)
            subscription_id = self.broker.subscribe_realtime_data(
                contract=contract,
                bar_size=bar_size,
                data_type=data_type,
                callback=self._handle_market_data_callback,
            )
            self._data_subscription_ids.append(subscription_id)

    def _unsubscribe_market_data(self) -> None:
        for subscription_id in list(self._data_subscription_ids):
            self.broker.unsubscribe_realtime_data(subscription_id)
        self._data_subscription_ids.clear()

    def _handle_market_data_callback(self, bar_data: BarData) -> None:
        if self._loop is None or self._loop.is_closed():
            logger.warning(
                "Dropping market bar for session %s because no event loop is active",
                self.session_id,
            )
            return
        future = asyncio.run_coroutine_threadsafe(
            self.process_market_data(bar_data),
            self._loop,
        )
        future.add_done_callback(self._log_callback_result)

    def _log_callback_result(
        self, future: asyncio.Future[TradingDecision | None]
    ) -> None:
        try:
            future.result()
        except Exception:
            logger.exception(
                "Async market-data callback failed for %s", self.session_id
            )

    def _on_order_update(self, order_status: OrderStatus) -> None:
        self.state.update_order_status(order_status.order_id, order_status.status)
        self.trading_metrics.record_order_status(self.session_id, order_status.status)
        order = self.state.orders.get(order_status.order_id)
        if order is None or order_status.status != "FILLED":
            return
        fill_price = float(order_status.average_fill_price or order.price)
        slippage = fill_price - order.price
        self.trading_metrics.record_fill(
            self.session_id,
            order.action,
            float(order_status.filled_quantity),
            fill_price,
            slippage,
        )

    def _on_position_update(self, position: PositionInfo) -> None:
        self.state.set_position(position.contract.symbol, float(position.quantity))
        self.trading_metrics.update_position(
            self.session_id,
            position.contract.symbol,
            float(position.quantity),
            float(position.market_value or 0.0),
        )
        self.trading_metrics.update_pnl(
            self.session_id,
            0.0,
            float(position.unrealized_pnl or 0.0),
        )

    async def _execute_decision(self, decision: TradingDecision) -> None:
        symbol = self._decision_symbol(decision)
        contract = self._symbol_to_contract(symbol)
        action = decision.action

        if action == "BUY":
            quantity = self._calculate_position_size(decision)
            if quantity <= 0:
                logger.info("Skipping BUY with zero calculated quantity for %s", symbol)
                return
            order_id = self.broker.place_order(
                contract,
                OrderSpec(
                    side=OrderSide.BUY,
                    quantity=quantity,
                    order_type=OrderType.MARKET,
                ),
            )
            self.state.record_order(
                order_id=order_id,
                symbol=symbol,
                action="BUY",
                quantity=quantity,
                decision=decision,
            )
            self.trading_metrics.record_order(
                self.session_id,
                "BUY",
                quantity,
                decision.market_price,
            )
            self.trading_metrics.update_position(
                self.session_id,
                symbol,
                self.state.get_position(symbol),
                self.state.get_position(symbol) * decision.market_price,
            )
            await self._persist_state()
            return

        if action == "SELL":
            current_position = self.state.get_position(symbol)
            allow_short = bool(self.config.risk_config.get("allow_short", False))
            quantity = current_position if current_position > 0 else 0.0
            if quantity <= 0 and allow_short:
                quantity = self._calculate_position_size(decision)
            if quantity <= 0:
                logger.info("Skipping SELL with no long position for %s", symbol)
                return
            order_id = self.broker.place_order(
                contract,
                OrderSpec(
                    side=OrderSide.SELL,
                    quantity=quantity,
                    order_type=OrderType.MARKET,
                ),
            )
            self.state.record_order(
                order_id=order_id,
                symbol=symbol,
                action="SELL",
                quantity=quantity,
                decision=decision,
            )
            self.trading_metrics.record_order(
                self.session_id,
                "SELL",
                quantity,
                decision.market_price,
            )
            self.trading_metrics.update_position(
                self.session_id,
                symbol,
                self.state.get_position(symbol),
                self.state.get_position(symbol) * decision.market_price,
            )
            await self._persist_state()

    async def _close_all_positions(self) -> None:
        for symbol, quantity in list(self.state.positions.items()):
            if abs(quantity) < 1e-9:
                continue
            action = "SELL" if quantity > 0 else "BUY"
            decision = TradingDecision(
                timestamp=datetime.now(tz=UTC),
                action=action,
                confidence=1.0,
                market_price=max(float(self.state.last_price or 0.0), 0.0),
                signals_used=[],
                latency_ms=0.0,
                metadata={"reason": "session_stop_close_positions"},
            )
            order_id = self.broker.place_order(
                self._symbol_to_contract(symbol),
                OrderSpec(
                    side=OrderSide.BUY if action == "BUY" else OrderSide.SELL,
                    quantity=abs(quantity),
                    order_type=OrderType.MARKET,
                ),
            )
            self.state.record_order(
                order_id=order_id,
                symbol=symbol,
                action=action,
                quantity=abs(quantity),
                decision=decision,
            )
            self.trading_metrics.record_order(
                self.session_id,
                action,
                abs(quantity),
                decision.market_price,
            )
            self.trading_metrics.update_position(
                self.session_id,
                symbol,
                self.state.get_position(symbol),
                self.state.get_position(symbol) * decision.market_price,
            )

    def _symbol_to_contract(self, symbol: str) -> ContractSpec:
        return ContractSpec(
            symbol=symbol.strip().upper(),
            instrument_type=InstrumentType.STOCK,
            exchange=str(self.config.risk_config.get("exchange", "SMART")),
            currency=str(self.config.risk_config.get("currency", "USD")),
            primary_exchange=self._optional_risk_value("primary_exchange"),
        )

    def _calculate_position_size(self, decision: TradingDecision) -> float:
        fixed_quantity = self.config.risk_config.get("fixed_quantity")
        if fixed_quantity is not None:
            return max(0.0, float(fixed_quantity))

        if decision.market_price <= 0:
            return 0.0

        account = self.broker.get_account_info()
        max_position_pct = float(self.config.risk_config.get("max_position_pct", 0.1))
        if max_position_pct <= 0 or not isfinite(max_position_pct):
            return 0.0

        max_position_value = account.buying_power * max_position_pct
        quantity = floor(max_position_value / decision.market_price)
        max_quantity = self.config.risk_config.get("max_quantity")
        if max_quantity is not None:
            quantity = min(quantity, int(max_quantity))
        return float(max(0, quantity))

    def _decision_symbol(self, decision: TradingDecision) -> str:
        raw_symbol = decision.metadata.get("symbol")
        if (
            isinstance(raw_symbol, str)
            and raw_symbol.strip().upper() in self.config.symbols
        ):
            return raw_symbol.strip().upper()
        return self.config.symbols[0]

    def _min_confidence(self) -> float:
        value = float(self.config.risk_config.get("min_confidence", 0.6))
        if not isfinite(value):
            return 0.6
        return max(0.0, min(1.0, value))

    def _optional_risk_value(self, key: str) -> str | None:
        value = self.config.risk_config.get(key)
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    async def _persist_state(self) -> None:
        await self.persistence.save_session(self.session_id, self.to_record())


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return _ensure_aware(parsed)


def _parse_optional_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    return _parse_datetime(str(value))


def _ensure_optional_aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return _ensure_aware(value)


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value


__all__ = [
    "InferencePipelineProtocol",
    "SessionConfig",
    "SessionStatus",
    "TradingSession",
]
