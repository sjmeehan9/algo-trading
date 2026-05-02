"""Trading session manager and production factory helpers."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from uuid import uuid4

from algotrading.src.broker import BrokerAdapter, BrokerConfig, BrokerRegistry
from algotrading.src.models.inference import SignalCache, TimestampAlignmentService
from algotrading.src.models.registry import SupportingModelRegistry
from algotrading.src.trading.deployment import DeploymentValidator
from algotrading.src.trading.inference import (
    LatencyTracker,
    RealTimeInferencePipeline,
    SignalAggregator,
)
from algotrading.src.trading.session.exceptions import (
    SessionConfigurationError,
    SessionNotFoundError,
)
from algotrading.src.trading.session.persistence import SessionPersistence
from algotrading.src.trading.session.session import (
    InferencePipelineProtocol,
    SessionConfig,
    SessionStatus,
    TradingSession,
)
from algotrading.src.trainers import RLTrainer
from algotrading.src.trainers.sb3_trainer import SB3Algorithm, StableBaselines3Trainer

logger = logging.getLogger(__name__)

BrokerFactory = Callable[[str], BrokerAdapter]
PipelineFactory = Callable[
    [SessionConfig], InferencePipelineProtocol | Awaitable[InferencePipelineProtocol]
]


class TradingSessionManager:
    """Coordinate creation, recovery, and lifecycle of trading sessions."""

    def __init__(
        self,
        broker_registry: BrokerRegistry,
        model_service: object,
        deployment_validator: DeploymentValidator,
        persistence: SessionPersistence,
        *,
        broker_config: BrokerConfig | Mapping[str, object] | None = None,
        broker_factory: BrokerFactory | None = None,
        pipeline_factory: PipelineFactory | None = None,
        supporting_registry: SupportingModelRegistry | None = None,
    ) -> None:
        """Initialize manager dependencies.

        Args:
            broker_registry: Registry used to create broker adapters.
            model_service: Model metadata service used for loading generations.
            deployment_validator: RL-only deployment validator.
            persistence: Session persistence layer.
            broker_config: Optional registry configuration override.
            broker_factory: Optional test seam for broker creation.
            pipeline_factory: Optional test seam for inference pipeline creation.
            supporting_registry: Optional supporting model registry override.
        """

        self.broker_registry = broker_registry
        self.model_service = model_service
        self.deployment_validator = deployment_validator
        self.persistence = persistence
        self.broker_config = broker_config
        self.broker_factory = broker_factory
        self.pipeline_factory = pipeline_factory
        self.supporting_registry = supporting_registry or getattr(
            model_service,
            "supporting_registry",
            SupportingModelRegistry(),
        )

        self._sessions: dict[str, TradingSession] = {}
        self._lock = asyncio.Lock()
        self._initialized = False
        self._initialize_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Load persisted sessions and recover active ones in paused state."""

        async with self._initialize_lock:
            if self._initialized:
                return

            persisted_sessions = await self.persistence.load_all_sessions()
            for session_data in persisted_sessions:
                try:
                    await self._recover_session(session_data)
                except Exception:
                    logger.exception(
                        "Failed to recover trading session %s",
                        session_data.get("session_id"),
                    )
            self._initialized = True

    async def create_session(
        self,
        *,
        model_id: str,
        generation_id: str,
        mode: str,
        broker: str,
        symbols: list[str],
        supporting_model_ids: list[str] | None = None,
        risk_config: dict[str, object] | None = None,
        user_id: str | None = None,
    ) -> TradingSession:
        """Create and persist a new trading session in ``created`` state."""

        await self.initialize()
        config = SessionConfig(
            model_id=model_id,
            generation_id=generation_id,
            broker_name=broker,
            mode=mode,
            symbols=symbols,
            supporting_model_ids=supporting_model_ids or [],
            risk_config=risk_config or {},
        )

        self.deployment_validator.validate_deployment(
            model_id=config.model_id,
            user_id=user_id,
            deployment_mode=config.mode,
            generation_id=config.generation_id,
        )

        broker_adapter = self._create_broker(config.broker_name)
        inference_pipeline = await self._build_inference_pipeline(config)
        session = TradingSession(
            session_id=str(uuid4()),
            config=config,
            broker=broker_adapter,
            inference_pipeline=inference_pipeline,
            persistence=self.persistence,
        )

        async with self._lock:
            self._sessions[session.session_id] = session
        await self.persistence.save_session(session.session_id, session.to_record())
        logger.info("Created trading session %s", session.session_id)
        return session

    async def get_session(self, session_id: str) -> TradingSession:
        """Return a session by ID."""

        await self.initialize()
        async with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        return session

    async def list_sessions(
        self,
        status: SessionStatus | None = None,
    ) -> list[TradingSession]:
        """List sessions, optionally filtered by lifecycle status."""

        await self.initialize()
        async with self._lock:
            sessions = list(self._sessions.values())
        if status is not None:
            sessions = [session for session in sessions if session.status == status]
        sessions.sort(key=lambda session: session.created_at, reverse=True)
        return sessions

    async def start_session(self, session_id: str) -> TradingSession:
        """Start a trading session and return the updated session."""

        session = await self.get_session(session_id)
        await session.start()
        return session

    async def pause_session(self, session_id: str) -> TradingSession:
        """Pause a running trading session."""

        session = await self.get_session(session_id)
        await session.pause()
        return session

    async def stop_session(
        self,
        session_id: str,
        close_positions: bool = False,
    ) -> TradingSession:
        """Stop a trading session and optionally close positions."""

        session = await self.get_session(session_id)
        await session.stop(close_positions=close_positions)
        return session

    async def shutdown(self, close_positions: bool = False) -> None:
        """Gracefully stop all running or paused sessions."""

        await self.initialize()
        sessions = await self.list_sessions()
        for session in sessions:
            if session.status in {SessionStatus.RUNNING, SessionStatus.PAUSED}:
                try:
                    await session.stop(close_positions=close_positions)
                except Exception:
                    logger.exception("Error stopping session %s", session.session_id)

    async def _recover_session(self, session_data: dict[str, object]) -> None:
        config = SessionConfig.from_dict(dict(session_data["config"]))
        broker_adapter = self._create_broker(config.broker_name)
        inference_pipeline = await self._build_inference_pipeline(config)
        session = TradingSession.from_record(
            session_data,
            broker=broker_adapter,
            inference_pipeline=inference_pipeline,
            persistence=self.persistence,
        )
        async with self._lock:
            self._sessions[session.session_id] = session
        await self.persistence.save_session(session.session_id, session.to_record())
        logger.info(
            "Recovered trading session %s in %s state",
            session.session_id,
            session.status.value,
        )

    def _create_broker(self, broker_name: str) -> BrokerAdapter:
        if self.broker_factory is not None:
            broker = self.broker_factory(broker_name)
            if not broker.is_connected():
                broker.connect(host=broker_name, port=0, client_id=0)
            return broker

        return self.broker_registry.create_broker(
            broker_name,
            config=self.broker_config,
        )

    async def _build_inference_pipeline(
        self,
        config: SessionConfig,
    ) -> InferencePipelineProtocol:
        if self.pipeline_factory is not None:
            pipeline = self.pipeline_factory(config)
            if inspect.isawaitable(pipeline):
                return await pipeline
            return pipeline

        core_model = self._load_core_model(config)
        signal_cache = SignalCache()
        alignment_service = TimestampAlignmentService(signal_cache)
        signal_aggregator = SignalAggregator(alignment_service)
        for model_id in config.supporting_model_ids:
            signal_aggregator.register_signal_default(model_id, 0.0)

        return RealTimeInferencePipeline(
            core_model=core_model,
            supporting_registry=self.supporting_registry,
            signal_aggregator=signal_aggregator,
            latency_tracker=LatencyTracker(),
            config={
                "model_type": "core_rl",
                "symbol": config.symbols[0],
                "supporting_model_ids": list(config.supporting_model_ids),
            },
        )

    def _load_core_model(self, config: SessionConfig) -> RLTrainer:
        model = self.model_service.get_model(config.model_id)
        generation = self.model_service.get_generation_detail(
            config.model_id,
            config.generation_id,
        )
        model_path = getattr(generation, "model_path", None)
        if not model_path:
            raise SessionConfigurationError(
                "Selected generation has no model artifact path"
            )

        algorithm_value = str(getattr(model, "algorithm", "")).strip().lower()
        try:
            algorithm = SB3Algorithm(algorithm_value)
        except ValueError as exc:
            raise SessionConfigurationError(
                f"Unsupported core RL algorithm for live trading: {algorithm_value}"
            ) from exc

        trainer = StableBaselines3Trainer(algorithm)
        trainer.load(str(model_path))
        return trainer


def create_default_session_manager(
    *,
    model_service: object,
    deployment_validator: DeploymentValidator,
    project_root: Path | None = None,
) -> TradingSessionManager:
    """Build a production session manager with filesystem persistence."""

    root = project_root or Path(__file__).resolve().parents[5]
    persistence = SessionPersistence(root / "data" / "sessions")
    return TradingSessionManager(
        broker_registry=BrokerRegistry(),
        model_service=model_service,
        deployment_validator=deployment_validator,
        persistence=persistence,
        supporting_registry=getattr(model_service, "supporting_registry", None),
    )


__all__ = [
    "BrokerFactory",
    "PipelineFactory",
    "TradingSessionManager",
    "create_default_session_manager",
]
