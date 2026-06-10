"""Phase 7.6 provider-gated regression for historical market acquisition.

These tests exercise the real broker acquisition path through the production
``DataAcquisitionService`` / ``HistoricalMarketDataAcquirer`` and prove that
live vendor bars are persisted into the canonical local store with a manifest
recording provider, range, and row counts.

Provider gating (see ``tests/conftest.py`` and ``tests/integration/conftest.py``):

* The Alpaca test is marked ``requires_alpaca`` and only runs with
  ``--alpaca-confirm``; it skips cleanly otherwise and skips if the Alpaca
  credentials are not set in the environment.
* The IB test is marked ``requires_ib`` and only runs with ``--ib-confirm``
  against a running TWS/Gateway; it skips cleanly otherwise.

No secrets are printed by these tests. Credentials come from the existing
confirm fixtures, which read them from the environment.

Run examples::

    pytest tests/integration/test_data_acquisition_market.py --alpaca-confirm -s -q
    pytest tests/integration/test_data_acquisition_market.py --ib-confirm -s -q
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from algotrading.api.schemas.data_sources import (
    MarketDataProvider,
    normalize_training_data_request,
)
from algotrading.src.broker.registry import BrokerRegistry
from algotrading.src.data_pipeline.acquisition import (
    AcquisitionStatus,
    HistoricalMarketDataAcquirer,
)
from algotrading.src.data_pipeline.storage import LocalDataStore

# A known regular-session window keeps provider-confirm tests stable after
# market close and on weekends; vendors return the historical bars regardless.
_REGULAR_SESSION_START = datetime(2026, 2, 13, 15, 0, tzinfo=UTC)


def _assert_canonical_market_artifacts(
    *,
    store: LocalDataStore,
    request: object,
    expected_provider: str,
    symbol: str,
    row_count: int,
    data_path: str | None,
    manifest_path: str | None,
) -> None:
    """Assert canonical market files, manifest, and loaded data are consistent."""

    assert row_count > 0, "provider returned no historical bars"
    assert data_path and Path(data_path).exists(), "canonical market.csv missing"
    assert (
        manifest_path and Path(manifest_path).exists()
    ), "canonical manifest.json missing"

    # The loaded canonical data round-trips and matches the reported row count.
    loaded = store.find_market_data(request)
    stored = loaded[symbol]
    assert stored.manifest.provider == expected_provider
    assert stored.manifest.record_count == row_count
    assert stored.manifest.symbols == (symbol,)
    # The manifest records an actual coverage range from the live bars.
    assert stored.manifest.actual_start is not None
    assert stored.manifest.actual_end is not None
    assert stored.manifest.actual_start <= stored.manifest.actual_end
    # The most recent bar carries a positive close price.
    assert stored.batch.records[-1].payload["close"] > 0


@pytest.mark.requires_alpaca
@pytest.mark.slow
def test_market_acquisition_alpaca_persists_canonical_store(
    confirm_alpaca_paper: dict[str, str],
    tmp_path: Path,
) -> None:
    """Acquire live Alpaca bars and persist canonical market data + manifest."""

    pytest.importorskip("alpaca")

    # Alpaca free-tier data lags realtime; request a closed historical window.
    end = datetime.now(tz=UTC) - timedelta(minutes=20)
    start = end - timedelta(days=5)
    request = normalize_training_data_request(
        {
            "symbols": ["AAPL"],
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "data_frequency": "1m",
            "cache_policy": "refresh",
            "market": {
                "provider": "alpaca",
                "bar_size": "1 min",
                "broker_data_type": "TRADES",
                "exchange": "SMART",
                "currency": "USD",
                "primary_exchange": "NASDAQ",
            },
        }
    )
    store = LocalDataStore(root=tmp_path / "sourced")
    acquirer = HistoricalMarketDataAcquirer(
        store=store,
        broker_registry=BrokerRegistry.isolated(),
        broker_config={**confirm_alpaca_paper, "paper": True, "data_feed": "iex"},
        broker_connection_params={},
    )

    result = acquirer.acquire(request)

    assert result.provider == MarketDataProvider.ALPACA.value
    report = result.reports[0]
    assert report.provider == MarketDataProvider.ALPACA.value
    assert report.status == AcquisitionStatus.ACQUIRED.value
    _assert_canonical_market_artifacts(
        store=store,
        request=request,
        expected_provider=MarketDataProvider.ALPACA.value,
        symbol="AAPL",
        row_count=report.row_count,
        data_path=report.data_path,
        manifest_path=report.manifest_path,
    )


@pytest.mark.requires_ib
@pytest.mark.slow
def test_market_acquisition_ib_persists_canonical_store(
    confirm_ib_gateway: dict[str, object],
    tmp_path: Path,
) -> None:
    """Acquire live IB bars from a running TWS and persist canonical market data."""

    conn = confirm_ib_gateway
    start = _REGULAR_SESSION_START
    end = start + timedelta(minutes=5)
    request = normalize_training_data_request(
        {
            "symbols": ["AMD"],
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "data_frequency": "5s",
            "cache_policy": "refresh",
            "market": {
                "provider": "ib",
                "bar_size": "5 secs",
                "broker_data_type": "TRADES",
                "exchange": "SMART",
                "currency": "USD",
            },
        }
    )
    store = LocalDataStore(root=tmp_path / "sourced")
    acquirer = HistoricalMarketDataAcquirer(
        store=store,
        broker_registry=BrokerRegistry.isolated(),
        broker_config={},
        broker_connection_params={
            "host": str(conn["host"]),
            "port": int(conn["port"]),
            # Offset the client id to avoid colliding with other IB sessions.
            "client_id": int(conn["client_id"]) + 64,
        },
    )

    result = acquirer.acquire(request)

    assert result.provider == MarketDataProvider.IB.value
    report = result.reports[0]
    assert report.provider == MarketDataProvider.IB.value
    assert report.status == AcquisitionStatus.ACQUIRED.value
    _assert_canonical_market_artifacts(
        store=store,
        request=request,
        expected_provider=MarketDataProvider.IB.value,
        symbol="AMD",
        row_count=report.row_count,
        data_path=report.data_path,
        manifest_path=report.manifest_path,
    )
