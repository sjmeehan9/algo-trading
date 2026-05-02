"""Request and response schemas for trading session management."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TradingSessionCreateRequest(BaseModel):
    """Payload for creating a trading session."""

    model_id: str = Field(min_length=1)
    generation_id: str = Field(min_length=1)
    mode: Literal["paper", "live"] = "paper"
    broker: str = Field(default="interactive_brokers", min_length=1)
    symbols: list[str] = Field(min_length=1)
    supporting_model_ids: list[str] = Field(default_factory=list)
    risk_config: dict[str, object] = Field(default_factory=dict)

    @field_validator("broker")
    @classmethod
    def normalize_broker(cls, value: str) -> str:
        """Normalize broker names for registry lookup."""

        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("broker must be non-empty")
        return normalized

    @field_validator("symbols")
    @classmethod
    def normalize_symbols(cls, value: list[str]) -> list[str]:
        """Normalize and validate session symbols."""

        symbols = [symbol.strip().upper() for symbol in value if symbol.strip()]
        if not symbols:
            raise ValueError("at least one symbol is required")
        return symbols

    @field_validator("supporting_model_ids")
    @classmethod
    def normalize_supporting_model_ids(cls, value: list[str]) -> list[str]:
        """Normalize supporting model identifiers."""

        return [model_id.strip() for model_id in value if model_id.strip()]


class TradingSessionStopRequest(BaseModel):
    """Payload for stopping a trading session."""

    close_positions: bool = False


class TradingSessionStatus(BaseModel):
    """Serializable session status returned by trading endpoints."""

    session_id: str
    status: str
    model_id: str
    generation_id: str
    broker: str
    mode: str
    symbols: list[str]
    supporting_model_ids: list[str] = Field(default_factory=list)
    created_at: datetime
    started_at: datetime | None = None
    stopped_at: datetime | None = None
    positions: dict[str, float] = Field(default_factory=dict)
    orders_count: int = 0
    decisions_count: int = 0
    executed_orders_count: int = 0
    error_count: int = 0
    error_message: str | None = None
    last_price: float | None = None
    last_decision: dict[str, object] | None = None


__all__ = [
    "TradingSessionCreateRequest",
    "TradingSessionStatus",
    "TradingSessionStopRequest",
]
