"""Common response schemas for API endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

ResponseDataT = TypeVar("ResponseDataT")


class APIResponse(BaseModel, Generic[ResponseDataT]):
    """Standard success response envelope."""

    success: bool = True
    data: ResponseDataT
    message: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class APIError(BaseModel):
    """Standard error response envelope."""

    success: bool = False
    error: str
    error_code: str
    details: dict[str, Any] | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PaginatedResponse(BaseModel, Generic[ResponseDataT]):
    """Paginated response payload used by list endpoints."""

    items: list[ResponseDataT]
    total: int
    page: int
    page_size: int
    pages: int
