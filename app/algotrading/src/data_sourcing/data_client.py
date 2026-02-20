"""Factory helpers for data sourcing clients.

This module centralizes creation of historical and live data clients and
supports optional broker adapter injection for testing and future broker
implementations.
"""

from __future__ import annotations

from algotrading.src.broker import BrokerAdapter
from algotrading.src.data_sourcing.live_streaming import LiveData
from algotrading.src.data_sourcing.save_historical import PastData


def create_historical_client(
    config: dict,
    pipeline: dict,
    adapter: BrokerAdapter | None = None,
) -> PastData:
    """Create a historical data client.

    Args:
        config: Runtime configuration dictionary.
        pipeline: Pipeline configuration dictionary.
        adapter: Optional broker adapter instance.

    Returns:
        Configured `PastData` client.
    """

    return PastData(config=config, pipeline=pipeline, adapter=adapter)


def create_live_client(
    config: dict,
    pipeline: dict,
    adapter: BrokerAdapter | None = None,
) -> LiveData:
    """Create a live streaming data client.

    Args:
        config: Runtime configuration dictionary.
        pipeline: Pipeline configuration dictionary.
        adapter: Optional broker adapter instance.

    Returns:
        Configured `LiveData` client.
    """

    return LiveData(config=config, pipeline=pipeline, adapter=adapter)
