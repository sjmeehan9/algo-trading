"""FastAPI application factory and top-level API endpoints."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from algotrading.api.config import APIConfig, as_public_config
from algotrading.api.middleware import AuthMiddleware, RequestLoggingMiddleware
from algotrading.api.schemas import APIError
from algotrading.api.websocket import WebSocketManager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(_, exc: StarletteHTTPException) -> JSONResponse:
        return _error_response(
            status_code=exc.status_code,
            error=str(exc.detail),
            error_code="HTTP_ERROR",
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

    @app.get("/health", tags=["system"])
    async def health_check() -> dict[str, str]:
        """Return service health state and timestamp."""

        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

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
