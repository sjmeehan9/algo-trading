"""Integration tests for BrokerDataSource with live Interactive Brokers adapter."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from algotrading.src.broker import InteractiveBrokersAdapter
from algotrading.src.data_pipeline.sources.broker_source import BrokerDataSource


@pytest.mark.requires_ib
def test_broker_source_fetch_batch_matches_adapter_bar_count(
    confirm_ib_gateway: dict,
) -> None:
    """BrokerDataSource historical batch aligns with direct adapter retrieval."""

    conn = confirm_ib_gateway
    adapter = InteractiveBrokersAdapter()
    source = BrokerDataSource(adapter=adapter)

    try:
        adapter.connect(conn["host"], conn["port"], conn["client_id"] + 10)
        assert source.is_connected

        end = datetime.now(tz=UTC)
        start = end - timedelta(minutes=15)

        batch = source.fetch_batch(symbol="AMD", start=start, end=end)

        assert len(batch.records) > 0
        assert batch.symbol == "AMD"
        assert batch.records[-1].payload["close"] > 0
        assert {
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vwap",
            "trade_count",
        }.issubset(set(batch.records[-1].payload.keys()))
    finally:
        adapter.disconnect()
