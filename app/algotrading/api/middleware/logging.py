"""Request/response logging middleware for the API service."""

from __future__ import annotations

import logging
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log request metadata and response timing for each HTTP request."""

    async def dispatch(self, request: Request, call_next):
        """Log request start and completion details."""

        start = time.perf_counter()
        client_ip = request.client.host if request.client else "unknown"

        logger.info(
            "request_started method=%s path=%s client_ip=%s",
            request.method,
            request.url.path,
            client_ip,
        )

        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.exception(
                "request_failed method=%s path=%s client_ip=%s duration_ms=%.2f",
                request.method,
                request.url.path,
                client_ip,
                elapsed_ms,
            )
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "request_completed method=%s path=%s client_ip=%s status=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            client_ip,
            response.status_code,
            elapsed_ms,
        )
        return response
