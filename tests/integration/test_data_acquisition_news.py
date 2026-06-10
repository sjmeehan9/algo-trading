"""Phase 7.6 provider-gated regression for historical news acquisition.

This test exercises the real news provider path through the production
``HistoricalNewsAcquirer`` and proves that live provider headlines are
persisted into the canonical local store with a manifest recording provider,
range, and record counts.

Provider gating (see ``tests/conftest.py`` and ``tests/integration/conftest.py``):

* The test is marked ``requires_news_api`` and only runs with
  ``--news-api-confirm``; it skips cleanly otherwise and skips if the
  ``BENZINGA_API_KEY`` / ``ALPHAVANTAGE_API_KEY`` environment variables are not
  set.

The ``BenzingaNewsSource`` disables SDK endpoint logging (Component 7.3), so no
URL containing an API token is printed. The configured provider is tried first,
followed by the configured fallback.

Run example::

    pytest tests/integration/test_data_acquisition_news.py --news-api-confirm -s -q
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from algotrading.api.schemas.data_sources import normalize_training_data_request
from algotrading.src.data_pipeline.acquisition import (
    AcquisitionStatus,
    HistoricalNewsAcquirer,
)
from algotrading.src.data_pipeline.sources.news_factory import NewsSourceFactory
from algotrading.src.data_pipeline.storage import LocalDataStore


@pytest.mark.requires_news_api
@pytest.mark.slow
def test_news_acquisition_provider_persists_canonical_store(
    confirm_news_api: dict[str, str],
    tmp_path: Path,
) -> None:
    """Acquire real provider news and persist canonical JSONL plus manifest.

    A fixed historical window is used instead of "the last 7 days". Provider
    news coverage is bounded by the API key's plan, and a relative recent window
    can land in a period the configured key does not serve, producing an empty
    (and therefore flaky) result that is unrelated to acquisition correctness.
    Early January 2025 is a stable window with established equity coverage, so
    the test reliably exercises the live provider round-trip and canonical
    persistence.
    """

    start = datetime(2025, 1, 2, tzinfo=UTC)
    end = datetime(2025, 1, 15, tzinfo=UTC)
    request = normalize_training_data_request(
        {
            "symbols": ["AAPL"],
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "data_frequency": "1d",
            "cache_policy": "refresh",
            "market": {"provider": "file"},
            "news": {
                "enabled": True,
                "required": True,
                "provider": "benzinga",
                "fallback_provider": "alphavantage",
                "include_body": False,
                "limit": 10,
            },
        }
    )
    factory = NewsSourceFactory.from_files(
        providers_path=confirm_news_api["providers_path"],
        credentials_path=confirm_news_api["credentials_path"],
    )
    store = LocalDataStore(root=tmp_path / "sourced")
    acquirer = HistoricalNewsAcquirer(store=store, news_source_factory=factory)

    result = acquirer.acquire(request)

    report = result.reports[0]
    assert report.symbol == "AAPL"
    assert report.status == AcquisitionStatus.ACQUIRED.value
    assert report.row_count > 0, "provider returned no historical news"
    assert report.data_path and Path(report.data_path).exists()
    assert report.manifest_path and Path(report.manifest_path).exists()

    # The canonical news round-trips and matches the reported record count.
    loaded = store.find_news_data(request)
    stored = loaded["AAPL"]
    assert stored.manifest.record_count == report.row_count
    assert stored.manifest.symbols == ("AAPL",)
    assert stored.manifest.actual_start is not None
    assert stored.manifest.actual_end is not None
    # Each persisted record carries a provider news id and a headline.
    assert stored.records[0].news_id
    assert stored.records[0].headline

    # The JSONL line count on disk matches the manifest record count.
    jsonl_lines = [
        line
        for line in Path(report.data_path).read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(jsonl_lines) == report.row_count
