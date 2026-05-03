"""FastAPI application factory and top-level API endpoints."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from algotrading.api.config import APIConfig, as_public_config
from algotrading.api.middleware import AuthMiddleware, RequestLoggingMiddleware
from algotrading.api.routers import (
    backtesting_router,
    deployment_router,
    generations_router,
    models_router,
    optimizer_router,
    strategies_router,
    trading_router,
    training_router,
)
from algotrading.api.schemas import APIError
from algotrading.api.websocket import WebSocketManager
from algotrading.src.broker import BrokerRegistry
from algotrading.src.config.validation import log_configuration, validate_configuration
from algotrading.src.monitoring import (
    HealthChecker,
    HealthStatus,
    MetricsCollector,
    setup_structured_logging,
)
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


def _error_response(
    status_code: int,
    error: str,
    error_code: str,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    """Build a standardized API error response."""

    payload = APIError(error=error, error_code=error_code, details=details)
    return JSONResponse(
        status_code=status_code, content=payload.model_dump(mode="json")
    )


def create_app(config: APIConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application instance.

    Args:
        config: Optional pre-resolved API configuration used by tests and runtime.

    Returns:
        Configured ``FastAPI`` application.
    """

    resolved_config = config or APIConfig()
    setup_structured_logging(level=resolved_config.log_level)

    is_valid, configuration_errors = validate_configuration(resolved_config)
    if not is_valid:
        for error in configuration_errors:
            logger.error("Configuration error: %s", error)
        raise RuntimeError("Invalid configuration: " + "; ".join(configuration_errors))
    log_configuration(resolved_config)

    app = FastAPI(
        title="Algo-Trading Model API",
        description="API for model configuration, training, and evaluation",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc" if resolved_config.debug else None,
    )

    app.state.api_config = resolved_config
    app.state.ws_manager = WebSocketManager()

    exempt_paths = (
        "/health",
        "/docs",
        "/docs/oauth2-redirect",
        "/redoc",
        "/openapi.json",
        "/metrics",
    )

    app.add_middleware(
        AuthMiddleware, api_key=resolved_config.api_key, exempt_paths=exempt_paths
    )
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(models_router, prefix="/api/v1")
    app.include_router(strategies_router, prefix="/api/v1")
    app.include_router(generations_router, prefix="/api/v1")
    app.include_router(training_router, prefix="/api/v1")
    app.include_router(backtesting_router, prefix="/api/v1")
    app.include_router(optimizer_router, prefix="/api/v1")
    app.include_router(deployment_router, prefix="/api/v1")
    app.include_router(trading_router, prefix="/api/v1")

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(_, exc: StarletteHTTPException) -> JSONResponse:
        details: dict[str, Any] | None = None
        error = str(exc.detail)
        error_code = "HTTP_ERROR"

        if isinstance(exc.detail, dict):
            error = str(exc.detail.get("error", "HTTP error"))
            error_code = str(exc.detail.get("error_code", "HTTP_ERROR"))
            details_value = exc.detail.get("details")
            if isinstance(details_value, dict):
                details = details_value

        return _error_response(
            status_code=exc.status_code,
            error=error,
            error_code=error_code,
            details=details,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_exception(
        _, exc: RequestValidationError
    ) -> JSONResponse:
        return _error_response(
            status_code=422,
            error="Request validation failed.",
            error_code="VALIDATION_ERROR",
            details={"errors": exc.errors()},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_exception(_, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception in API request.", exc_info=exc)
        return _error_response(
            status_code=500,
            error="Internal server error.",
            error_code="INTERNAL_SERVER_ERROR",
        )

    @app.on_event("startup")
    async def on_startup() -> None:
        logger.info("api_startup %s", as_public_config(resolved_config))
        training_service = getattr(app.state, "training_service", None)
        if training_service is not None:
            await training_service.start()

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        training_service = getattr(app.state, "training_service", None)
        if training_service is not None:
            await training_service.stop()
        session_manager = getattr(app.state, "trading_session_manager", None)
        if session_manager is not None:
            await session_manager.shutdown(close_positions=False)

    def get_health_checker() -> HealthChecker:
        """Return the configured health checker for this app instance."""

        checker = getattr(app.state, "health_checker", None)
        if checker is None:
            checker = HealthChecker(
                broker_registry=BrokerRegistry(),
                session_manager_provider=lambda: getattr(
                    app.state,
                    "trading_session_manager",
                    None,
                ),
            )
            app.state.health_checker = checker
        return checker

    @app.get("/health", tags=["system"])
    async def health_check(request: Request) -> JSONResponse:
        """Return detailed service health state and component statuses."""

        del request
        result = await get_health_checker().check_all()
        status_code = (
            503 if result.get("status") == HealthStatus.UNHEALTHY.value else 200
        )
        return JSONResponse(content=result, status_code=status_code)

    @app.get("/metrics", tags=["system"])
    async def metrics() -> PlainTextResponse:
        """Return Prometheus-compatible application metrics."""

        return PlainTextResponse(
            content=MetricsCollector().get_prometheus_format(),
            media_type="text/plain; version=0.0.4",
        )

    @app.get("/metrics/json", tags=["system"])
    async def metrics_json() -> dict[str, dict[str, object]]:
        """Return application metrics in JSON format for diagnostics."""

        return MetricsCollector().get_all()

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        """Handle WebSocket client connections and topic subscriptions."""

        provided_key = websocket.headers.get("X-API-Key") or websocket.query_params.get(
            "api_key"
        )
        if provided_key != resolved_config.api_key:
            await websocket.close(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="Invalid API key.",
            )
            return

        client_id = websocket.query_params.get("client_id") or str(uuid4())
        manager: WebSocketManager = app.state.ws_manager

        await manager.connect(websocket, client_id)
        await manager.send_to_client(
            client_id,
            {"type": "connected", "client_id": client_id},
        )

        try:
            while True:
                message = await websocket.receive_json()
                msg_type = message.get("type")
                topic = message.get("topic")

                if msg_type == "subscribe" and isinstance(topic, str):
                    manager.subscribe(client_id, topic)
                    await manager.send_to_client(
                        client_id,
                        {"type": "subscribed", "topic": topic},
                    )
                    continue

                if msg_type == "unsubscribe" and isinstance(topic, str):
                    manager.unsubscribe(client_id, topic)
                    await manager.send_to_client(
                        client_id,
                        {"type": "unsubscribed", "topic": topic},
                    )
                    continue

                if msg_type == "ping":
                    await manager.send_to_client(client_id, {"type": "pong"})
                    continue

                await manager.send_to_client(
                    client_id,
                    {
                        "type": "error",
                        "message": "Unsupported WebSocket message type.",
                    },
                )

        except WebSocketDisconnect:
            logger.info("websocket_disconnected client_id=%s", client_id)
        except Exception:
            logger.exception("websocket_error client_id=%s", client_id)
            try:
                await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
            except RuntimeError:
                logger.debug("WebSocket already closed for client_id=%s", client_id)
        finally:
            manager.disconnect(client_id)

    return app
