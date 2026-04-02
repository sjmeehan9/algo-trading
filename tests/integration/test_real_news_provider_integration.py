"""Human-gated integration tests for real news provider connectivity."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from algotrading.src.data_pipeline.sources import NewsSourceFactory
from algotrading.src.data_pipeline.types import NewsQuery


@pytest.mark.requires_news_api
@pytest.mark.slow
def test_primary_news_provider_connect_and_fetch(
    confirm_news_api: dict[str, str],
) -> None:
    """Primary configured provider should connect and return parsable records."""

    factory = NewsSourceFactory.from_files(
        providers_path=confirm_news_api["providers_path"],
        credentials_path=confirm_news_api["credentials_path"],
    )

    source = factory.create_primary()
    source.connect()
    try:
        end_time = datetime.now(tz=UTC)
        start_time = end_time - timedelta(days=2)
        records = source.fetch_news(
            NewsQuery(
                start_time=start_time,
                end_time=end_time,
                symbols=["AAPL"],
                limit=3,
                include_body=False,
            )
        )

        assert isinstance(records, list)
        if records:
            first = records[0]
            assert first.headline
            assert first.news_id
            assert first.timestamp.tzinfo is not None
    finally:
        source.disconnect()


@pytest.mark.requires_news_api
@pytest.mark.slow
def test_fallback_news_provider_connect_and_fetch(
    confirm_news_api: dict[str, str],
) -> None:
    """Fallback provider should connect and return valid record payloads."""

    factory = NewsSourceFactory.from_files(
        providers_path=confirm_news_api["providers_path"],
        credentials_path=confirm_news_api["credentials_path"],
    )

    fallback = factory.create_fallback()
    assert fallback is not None

    fallback.connect()
    try:
        end_time = datetime.now(tz=UTC)
        start_time = end_time - timedelta(days=2)
        records = fallback.fetch_news(
            NewsQuery(
                start_time=start_time,
                end_time=end_time,
                symbols=["AAPL"],
                limit=3,
                include_body=False,
            )
        )

        assert isinstance(records, list)
        if records:
            first = records[0]
            assert first.headline
            assert first.news_id
            assert first.timestamp.tzinfo is not None
    finally:
        fallback.disconnect()
