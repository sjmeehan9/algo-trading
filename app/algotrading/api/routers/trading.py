"""REST endpoints for live trading session lifecycle management."""

from __future__ import annotations

from algotrading.api.schemas.common import APIResponse
from algotrading.api.schemas.trading import (
    TradingSessionCreateRequest,
    TradingSessionStatus,
    TradingSessionStopRequest,
)
from algotrading.api.services.deployment_service import get_deployment_service
from algotrading.api.services.model_service import get_model_service
from algotrading.src.broker import BrokerConnectionError, NoBrokerAvailableError
from algotrading.src.trading.deployment import InvalidDeploymentError
from algotrading.src.trading.session import (
    InvalidSessionStateError,
    SessionConfigurationError,
    SessionNotFoundError,
    SessionStatus,
    TradingSessionManager,
    create_default_session_manager,
)
from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, status

router = APIRouter(prefix="/trading", tags=["trading"])


def _http_error(
    status_code: int,
    error: str,
    error_code: str,
    details: dict[str, object] | None = None,
) -> HTTPException:
    """Build a consistent HTTPException payload consumed by global handlers."""

    return HTTPException(
        status_code=status_code,
        detail={
            "error": error,
            "error_code": error_code,
            "details": details,
        },
    )


async def get_trading_session_manager(request: Request) -> TradingSessionManager:
    """FastAPI dependency resolver for the shared session manager."""

    from algotrading.api.services.app_state import get_or_create_state

    def _build() -> TradingSessionManager:
        model_service = get_model_service(request)
        deployment_service = get_deployment_service(request)
        return create_default_session_manager(
            model_service=model_service,
            deployment_validator=deployment_service.deployment_validator,
        )

    manager = get_or_create_state(request, "trading_session_manager", _build)
    await manager.initialize()
    return manager


@router.post(
    "/sessions",
    response_model=APIResponse[TradingSessionStatus],
    status_code=status.HTTP_201_CREATED,
)
async def create_session(
    payload: TradingSessionCreateRequest,
    request: Request,
    manager: TradingSessionManager = Depends(get_trading_session_manager),
) -> APIResponse[TradingSessionStatus]:
    """Create a trading session in created state."""

    user_id = request.headers.get("X-User") or request.headers.get("X-Client-ID")
    try:
        session = await manager.create_session(
            model_id=payload.model_id,
            generation_id=payload.generation_id,
            mode=payload.mode,
            broker=payload.broker,
            symbols=payload.symbols,
            supporting_model_ids=payload.supporting_model_ids,
            risk_config=payload.risk_config,
            user_id=user_id,
        )
    except InvalidDeploymentError as exc:
        raise _http_error(400, str(exc), "DEPLOYMENT_NOT_ALLOWED") from exc
    except (BrokerConnectionError, NoBrokerAvailableError) as exc:
        raise _http_error(503, str(exc), "BROKER_UNAVAILABLE") from exc
    except SessionConfigurationError as exc:
        raise _http_error(400, str(exc), "SESSION_CONFIGURATION_ERROR") from exc

    return APIResponse(
        data=TradingSessionStatus.model_validate(session.get_status()),
        message="Trading session created",
    )


@router.get("/sessions", response_model=APIResponse[list[TradingSessionStatus]])
async def list_sessions(
    status_filter: str | None = Query(default=None, alias="status"),
    manager: TradingSessionManager = Depends(get_trading_session_manager),
) -> APIResponse[list[TradingSessionStatus]]:
    """List trading sessions, optionally filtered by lifecycle status."""

    try:
        status_value = SessionStatus(status_filter) if status_filter else None
    except ValueError as exc:
        raise _http_error(
            400, "Unknown session status", "SESSION_STATUS_INVALID"
        ) from exc

    sessions = await manager.list_sessions(status=status_value)
    return APIResponse(
        data=[
            TradingSessionStatus.model_validate(session.get_status())
            for session in sessions
        ]
    )


@router.get(
    "/sessions/{session_id}",
    response_model=APIResponse[TradingSessionStatus],
)
async def get_session(
    session_id: str,
    manager: TradingSessionManager = Depends(get_trading_session_manager),
) -> APIResponse[TradingSessionStatus]:
    """Return details for one trading session."""

    try:
        session = await manager.get_session(session_id)
    except SessionNotFoundError as exc:
        raise _http_error(404, str(exc), "SESSION_NOT_FOUND") from exc
    return APIResponse(data=TradingSessionStatus.model_validate(session.get_status()))


@router.post(
    "/sessions/{session_id}/start",
    response_model=APIResponse[TradingSessionStatus],
)
async def start_session(
    session_id: str,
    manager: TradingSessionManager = Depends(get_trading_session_manager),
) -> APIResponse[TradingSessionStatus]:
    """Start or resume a trading session."""

    try:
        session = await manager.start_session(session_id)
    except SessionNotFoundError as exc:
        raise _http_error(404, str(exc), "SESSION_NOT_FOUND") from exc
    except InvalidSessionStateError as exc:
        raise _http_error(409, str(exc), "SESSION_STATE_ERROR") from exc
    return APIResponse(
        data=TradingSessionStatus.model_validate(session.get_status()),
        message="Trading session started",
    )


@router.post(
    "/sessions/{session_id}/pause",
    response_model=APIResponse[TradingSessionStatus],
)
async def pause_session(
    session_id: str,
    manager: TradingSessionManager = Depends(get_trading_session_manager),
) -> APIResponse[TradingSessionStatus]:
    """Pause a running trading session."""

    try:
        session = await manager.pause_session(session_id)
    except SessionNotFoundError as exc:
        raise _http_error(404, str(exc), "SESSION_NOT_FOUND") from exc
    except InvalidSessionStateError as exc:
        raise _http_error(409, str(exc), "SESSION_STATE_ERROR") from exc
    return APIResponse(
        data=TradingSessionStatus.model_validate(session.get_status()),
        message="Trading session paused",
    )


@router.post(
    "/sessions/{session_id}/stop",
    response_model=APIResponse[TradingSessionStatus],
)
async def stop_session(
    session_id: str,
    payload: TradingSessionStopRequest | None = Body(default=None),
    manager: TradingSessionManager = Depends(get_trading_session_manager),
) -> APIResponse[TradingSessionStatus]:
    """Stop a trading session."""

    close_positions = payload.close_positions if payload is not None else False
    try:
        session = await manager.stop_session(
            session_id,
            close_positions=close_positions,
        )
    except SessionNotFoundError as exc:
        raise _http_error(404, str(exc), "SESSION_NOT_FOUND") from exc
    except InvalidSessionStateError as exc:
        raise _http_error(409, str(exc), "SESSION_STATE_ERROR") from exc
    return APIResponse(
        data=TradingSessionStatus.model_validate(session.get_status()),
        message="Trading session stopped",
    )


__all__ = ["get_trading_session_manager", "router"]
