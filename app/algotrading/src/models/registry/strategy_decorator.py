"""Decorator and protocol definitions for custom strategy signal providers."""

from __future__ import annotations

from typing import Protocol

from algotrading.src.models.signals import ModelSignal, SignalType


class StrategyProtocol(Protocol):
    """Protocol for strategies that emit model signals."""

    def predict(self, state: dict[str, object], market_data: object) -> ModelSignal:
        """Produce a signal from the provided state and market data."""

    def reset(self) -> None:
        """Reset strategy internal state between episodes or sessions."""


def trading_strategy(
    name: str | None = None,
    signal_type: SignalType = SignalType.POSITION,
    description: str | None = None,
    version: str = "1.0.0",
):
    """Mark a class as a discoverable trading strategy.

    Args:
        name: Optional display name shown in strategy selection UIs.
        signal_type: Signal type emitted by the strategy.
        description: Optional description for operators and logs.
        version: Strategy semantic version label.

    Returns:
        A class decorator that attaches normalized strategy metadata.
    """

    def decorator(cls: type[object]) -> type[object]:
        cls_description = description or (cls.__doc__ or "")
        metadata = {
            "name": name or cls.__name__,
            "signal_type": signal_type,
            "description": cls_description.strip(),
            "version": version,
            "is_trading_strategy": True,
        }
        setattr(cls, "_strategy_metadata", metadata)
        return cls

    return decorator


__all__ = ["StrategyProtocol", "trading_strategy"]
