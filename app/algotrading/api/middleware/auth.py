"""Authentication middleware for API key protected endpoints."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class AuthMiddleware(BaseHTTPMiddleware):
    """Validate API key credentials for incoming HTTP requests."""

    def __init__(
        self,
        app,
        api_key: str,
        exempt_paths: Iterable[str] | None = None,
    ) -> None:
        super().__init__(app)
        self._api_key = api_key
        self._exempt_paths = tuple(exempt_paths or ())

    async def dispatch(self, request: Request, call_next):
        """Authorize requests unless their path is exempt."""

        if self._is_exempt(request.url.path):
            return await call_next(request)

        provided_key = request.headers.get("X-API-Key") or request.query_params.get(
            "api_key"
        )
        if not provided_key:
            return self._unauthorized(
                "API key is required.",
                "AUTH_MISSING_API_KEY",
            )

        if provided_key != self._api_key:
            return self._unauthorized(
                "Invalid API key.",
                "AUTH_INVALID_API_KEY",
            )

        request.state.authenticated = True
        return await call_next(request)

    def _is_exempt(self, path: str) -> bool:
        """Check whether a request path should bypass API key validation."""

        for exempt in self._exempt_paths:
            if path == exempt or path.startswith(f"{exempt}/"):
                return True
        return False

    def _unauthorized(self, error: str, error_code: str) -> JSONResponse:
        """Create the standard 401 response payload."""

        return JSONResponse(
            status_code=401,
            content={
                "success": False,
                "error": error,
                "error_code": error_code,
                "details": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
