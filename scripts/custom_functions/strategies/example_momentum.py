"""Example momentum strategy registered with the custom strategy registry."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
from algotrading.src.models.registry import trading_strategy
from algotrading.src.models.signals import ModelSignal, SignalMetadata, SignalType


@trading_strategy(
    name="Simple Momentum",
    signal_type=SignalType.POSITION,
    description="Buy when close is above MA, short when below.",
)
class SimpleMomentumStrategy:
    """Simple moving-average crossover strategy for demonstration."""

    def __init__(self, ma_period: int = 20) -> None:
        self.ma_period = ma_period

    def predict(
        self,
        state: dict[str, object],
        market_data: pd.DataFrame,
    ) -> ModelSignal:
        """Produce a position signal based on close vs moving average."""

        if market_data.empty or "close" not in market_data.columns:
            value = 0.0
            confidence = 0.0
        else:
            close_series = market_data["close"].astype(float)
            close = float(close_series.iloc[-1])
            ma = float(close_series.rolling(self.ma_period).mean().iloc[-1])

            if close > ma:
                value = 1.0
            elif close < ma:
                value = 2.0
            else:
                value = 0.0

            denominator = abs(ma) if abs(ma) > 1e-9 else 1.0
            confidence = min(1.0, abs(close - ma) / denominator)

        return ModelSignal(
            timestamp=datetime.now(tz=UTC),
            signal_type=SignalType.POSITION,
            value=value,
            confidence=confidence,
            metadata=SignalMetadata(
                model_id="simple_momentum",
                model_type="strategy",
                model_version="1.0.0",
            ),
            symbol=str(state.get("symbol")) if "symbol" in state else None,
        )

    def reset(self) -> None:
        """Reset strategy state."""

        return
