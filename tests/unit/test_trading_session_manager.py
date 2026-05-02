"""Unit tests for trading session lifecycle management."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest
from algotrading.src.broker import BarData, BrokerRegistry, ContractSpec, OrderSpec
from algotrading.src.trading.inference import TradingDecision
from algotrading.src.trading.session import (
    SessionConfig,
    SessionPersistence,
    SessionStatus,
    TradingSessionManager,
)

from tests.mocks import MockBrokerAdapter


class _Validator:
    """Deployment validator test double recording validation calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def validate_deployment(
        self,
        model_id: str,
        user_id: str | None = None,
        deployment_mode: str = "paper",
        generation_id: str | None = None,
        record_audit: bool = True,
    ) -> bool:
        """Record validation parameters and allow deployment."""

        del user_id, record_audit
        self.calls.append((model_id, generation_id or "", deployment_mode))
        return True


class _RecordingBroker(MockBrokerAdapter):
    """Mock broker that records order and unsubscribe operations."""

    def __init__(self) -> None:
        super().__init__()
        self.orders: list[tuple[ContractSpec, OrderSpec]] = []
        self.unsubscribed: list[int] = []

    def place_order(self, contract: ContractSpec, order: OrderSpec) -> str:
        """Record order submissions before delegating to the mock broker."""

        self.orders.append((contract, order))
        return super().place_order(contract, order)

    def unsubscribe_realtime_data(self, subscription_id: int) -> None:
        """Record subscription cleanup."""

        self.unsubscribed.append(subscription_id)
        super().unsubscribe_realtime_data(subscription_id)


class _DecisionPipeline:
    """Deterministic inference pipeline test double."""

    def __init__(self, decision_factory: Callable[[BarData], TradingDecision]) -> None:
        self.decision_factory = decision_factory
        self.started = False
        self.stopped = False
        self.processed: list[BarData] = []

    async def start(self, decision_callback=None) -> None:
        """Mark the pipeline as started."""

        del decision_callback
        self.started = True
        self.stopped = False

    async def stop(self) -> None:
        """Mark the pipeline as stopped."""

        self.stopped = True

    async def process_market_data(self, bar_data: BarData) -> TradingDecision:
        """Return a deterministic decision for a market bar."""

        self.processed.append(bar_data)
        return self.decision_factory(bar_data)


def _bar() -> BarData:
    """Build a valid broker bar for lifecycle tests."""

    return BarData(
        timestamp=datetime(2026, 5, 3, 14, 30, tzinfo=UTC),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        volume=10_000,
    )


def _decision(bar_data: BarData, action: str = "BUY") -> TradingDecision:
    """Build a decision tied to a broker bar."""

    return TradingDecision(
        timestamp=bar_data.timestamp,
        action=action,
        confidence=0.9,
        market_price=bar_data.close,
        signals_used=[],
        latency_ms=1.0,
    )


def _manager(
    tmp_path: Path,
    *,
    action: str = "BUY",
) -> tuple[TradingSessionManager, list[_RecordingBroker], _Validator]:
    """Build a session manager with deterministic broker and pipeline fakes."""

    brokers: list[_RecordingBroker] = []
    validator = _Validator()

    def broker_factory(broker_name: str) -> _RecordingBroker:
        del broker_name
        broker = _RecordingBroker()
        brokers.append(broker)
        return broker

    def pipeline_factory(config: SessionConfig) -> _DecisionPipeline:
        del config
        return _DecisionPipeline(lambda bar_data: _decision(bar_data, action=action))

    manager = TradingSessionManager(
        broker_registry=BrokerRegistry.isolated(include_builtins=False),
        model_service=object(),
        deployment_validator=validator,  # type: ignore[arg-type]
        persistence=SessionPersistence(tmp_path / "sessions"),
        broker_factory=broker_factory,
        pipeline_factory=pipeline_factory,
    )
    return manager, brokers, validator


@pytest.mark.asyncio
async def test_session_lifecycle_executes_buy_decision(tmp_path: Path) -> None:
    """A running session should subscribe, infer, place orders, and stop."""

    manager, brokers, validator = _manager(tmp_path)
    session = await manager.create_session(
        model_id="core-1",
        generation_id="gen-1",
        mode="paper",
        broker="mock",
        symbols=["AAPL"],
        risk_config={"max_position_pct": 0.01, "min_confidence": 0.5},
    )

    await manager.start_session(session.session_id)
    decision = await session.process_market_data(_bar())

    assert validator.calls == [("core-1", "gen-1", "paper")]
    assert session.status == SessionStatus.RUNNING
    assert decision is not None
    assert session.state.decisions_count == 1
    assert session.state.get_position("AAPL") == 20.0
    assert brokers[0].orders[0][1].side.value == "BUY"

    await manager.pause_session(session.session_id)
    assert session.status == SessionStatus.PAUSED

    await manager.start_session(session.session_id)
    assert session.status == SessionStatus.RUNNING

    await manager.stop_session(session.session_id)
    assert session.status == SessionStatus.STOPPED
    assert brokers[0].unsubscribed


@pytest.mark.asyncio
async def test_manager_recovers_running_session_as_paused(tmp_path: Path) -> None:
    """Persisted active sessions should recover conservatively as paused."""

    manager, _, _ = _manager(tmp_path)
    session = await manager.create_session(
        model_id="core-1",
        generation_id="gen-1",
        mode="paper",
        broker="mock",
        symbols=["AAPL"],
    )
    await manager.start_session(session.session_id)

    recovered_manager, _, _ = _manager(tmp_path)
    await recovered_manager.initialize()
    recovered_sessions = await recovered_manager.list_sessions()

    assert len(recovered_sessions) == 1
    assert recovered_sessions[0].session_id == session.session_id
    assert recovered_sessions[0].status == SessionStatus.PAUSED


@pytest.mark.asyncio
async def test_multiple_sessions_use_independent_brokers(tmp_path: Path) -> None:
    """Manager should support simultaneous sessions with separate adapters."""

    manager, brokers, _ = _manager(tmp_path)
    first = await manager.create_session(
        model_id="core-1",
        generation_id="gen-1",
        mode="paper",
        broker="mock-a",
        symbols=["AAPL"],
    )
    second = await manager.create_session(
        model_id="core-2",
        generation_id="gen-2",
        mode="paper",
        broker="mock-b",
        symbols=["MSFT"],
    )

    await manager.start_session(first.session_id)
    await manager.start_session(second.session_id)

    assert first.status == SessionStatus.RUNNING
    assert second.status == SessionStatus.RUNNING
    assert len(brokers) == 2
    assert brokers[0] is not brokers[1]


@pytest.mark.asyncio
async def test_shutdown_closes_positions_when_requested(tmp_path: Path) -> None:
    """Graceful shutdown can flatten tracked positions before stopping."""

    manager, brokers, _ = _manager(tmp_path, action="HOLD")
    session = await manager.create_session(
        model_id="core-1",
        generation_id="gen-1",
        mode="paper",
        broker="mock",
        symbols=["AAPL"],
    )
    await manager.start_session(session.session_id)
    session.state.set_position("AAPL", 5.0)

    await manager.shutdown(close_positions=True)

    assert session.status == SessionStatus.STOPPED
    assert brokers[0].orders[-1][1].side.value == "SELL"
    assert session.state.get_position("AAPL") == 0.0
