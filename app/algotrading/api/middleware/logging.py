"""Request/response logging middleware for the API service."""

from __future__ import annotations

import logging
import time
from uuid import uuid4

from algotrading.src.monitoring import (
    MetricsCollector,
    log_event,
    reset_correlation_id,
    set_correlation_id,
)
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log request metadata and response timing for each HTTP request."""

    async def dispatch(self, request: Request, call_next):
        """Log request start and completion details."""

        start = time.perf_counter()
        client_ip = request.client.host if request.client else "unknown"
        request_path = request.url.path
        correlation_value = request.headers.get("X-Correlation-ID") or str(uuid4())
        token = set_correlation_id(correlation_value)
        metrics = MetricsCollector()

        log_event(
            logger,
            "http.request.started",
            "request_started",
            method=request.method,
            path=request_path,
            client_ip=client_ip,
        )

        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            metrics.increment(
                "http_request_errors_total",
                tags={"method": request.method, "path": request_path},
            )
            metrics.record(
                "http_request_latency_ms",
                elapsed_ms,
                tags={
                    "method": request.method,
                    "path": request_path,
                    "status": "error",
                },
            )
            log_event(
                logger,
                "http.request.failed",
                "request_failed",
                level=logging.ERROR,
                method=request.method,
                path=request_path,
                client_ip=client_ip,
                duration_ms=round(elapsed_ms, 3),
                exc_info=True,
            )
            reset_correlation_id(token)
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        response.headers["X-Correlation-ID"] = correlation_value
        metrics.increment(
            "http_requests_total",
            tags={
                "method": request.method,
                "path": request_path,
                "status": response.status_code,
            },
        )
        metrics.record(
            "http_request_latency_ms",
            elapsed_ms,
            tags={
                "method": request.method,
                "path": request_path,
                "status": response.status_code,
            },
        )
        log_event(
            logger,
            "http.request.completed",
            "request_completed",
            method=request.method,
            path=request_path,
            client_ip=client_ip,
            status=response.status_code,
            duration_ms=round(elapsed_ms, 3),
        )
        reset_correlation_id(token)
        return response
