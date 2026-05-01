"""Request and response schemas for strategy endpoints."""

from __future__ import annotations

from algotrading.src.models.signals import SignalType
from pydantic import BaseModel


class StrategyInfo(BaseModel):
    """Strategy summary information."""

    strategy_id: str
    name: str
    signal_type: SignalType
    description: str
    version: str
    state: str = "ready"


class StrategyDetail(StrategyInfo):
    """Detailed strategy representation including source metadata."""

    filepath: str
    class_name: str


__all__ = ["StrategyInfo", "StrategyDetail"]
