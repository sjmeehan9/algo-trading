"""Benzinga-backed news source implementation for sentiment pipelines."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from itertools import count
from threading import Event, RLock, Thread
from typing import Any

from algotrading.src.data_pipeline.sources.exceptions import (
    DataSourceConnectionError,
    DataSourceError,
)
from algotrading.src.data_pipeline.sources.news_source import NewsDataSource
from algotrading.src.data_pipeline.sources.news_utils import (
    RateLimiter,
    generate_stable_id,
    parse_benzinga_datetime,
)
from algotrading.src.data_pipeline.types import (
    DataFrequency,
    DataType,
    NewsQuery,
    NewsRecord,
)

logger = logging.getLogger(__name__)


class BenzingaNewsSource(NewsDataSource):
    """Primary news source using the Benzinga Python SDK.

    Benzinga provides REST-style retrieval only. Real-time subscriptions are
    implemented with a polling loop using ``updated_since``.
    """

    def __init__(self, api_key: str, config: dict[str, object] | None = None) -> None:
        """Initialize source with API key and optional provider configuration."""

        if not api_key.strip():
            raise ValueError("api_key must be non-empty")

        self._api_key = api_key.strip()
        self._config = dict(config or {})
        self._source_id = str(self._config.get("source_id", "benzinga_news"))
        self._rate_limiter = RateLimiter(
            calls_per_minute=int(self._config.get("calls_per_minute", 100))
        )
        self._poll_interval_seconds = float(
            self._config.get("poll_interval_seconds", 10)
        )
        if self._poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be greater than 0")

        self._client: Any | None = None
        self._connected = False
        self._lock = RLock()
        self._subscription_ids = count(1)
        self._subscriptions: dict[int, tuple[Event, Thread]] = {}

    @property
    def source_id(self) -> str:
        """Return unique source identifier."""

        return self._source_id

    @property
    def is_connected(self) -> bool:
        """Return provider connection status."""

        return self._connected

    @property
    def metadata(self):
        """Return source capability metadata."""

        metadata = super().metadata
        metadata.config.update(
            {
                "provider": "benzinga",
                "poll_interval_seconds": self._poll_interval_seconds,
                "calls_per_minute": 100,
            }
        )
        return metadata

    def connect(self) -> None:
        """Initialize Benzinga SDK client and validate API access."""

        try:
            news_data = importlib.import_module("benzinga.news_data")
        except ImportError as exc:
            raise DataSourceConnectionError(
                "Benzinga SDK is not installed. Install package 'benzinga'."
            ) from exc

        try:
            client = news_data.News(self._api_key, log=False)
            _ = client.news(pagesize=1)
        except Exception as exc:  # pragma: no cover - network/provider behavior
            raise DataSourceConnectionError(
                f"Benzinga API connection failed: {exc}"
            ) from exc

        with self._lock:
            self._client = client
            self._connected = True

    def disconnect(self) -> None:
        """Stop active subscriptions and release provider resources."""

        with self._lock:
            subscription_ids = list(self._subscriptions.keys())

        for subscription_id in subscription_ids:
            self.unsubscribe_news(subscription_id)

        with self._lock:
            self._connected = False
            self._client = None

    def fetch_news(self, query: NewsQuery) -> list[NewsRecord]:
        """Fetch historical Benzinga news and map results to ``NewsRecord``."""

        client = self._require_client()
        self._rate_limiter.wait()

        params: dict[str, object] = {
            "company_tickers": ",".join(query.symbols) if query.symbols else None,
            "date_from": query.start_time.strftime("%Y-%m-%d"),
            "date_to": query.end_time.strftime("%Y-%m-%d"),
            "pagesize": min(query.limit, 1000),
            "display_output": "full" if query.include_body else "abstract",
        }
        if query.categories:
            params["channel"] = query.categories[0]

        clean_params = {
            key: value for key, value in params.items() if value is not None
        }

        try:
            raw_articles = client.news(**clean_params)
        except Exception as exc:  # pragma: no cover - network/provider behavior
            raise DataSourceError(f"Benzinga news request failed: {exc}") from exc

        articles = self._normalize_response(raw_articles)
        records = [
            self._parse_article(article=article, include_body=query.include_body)
            for article in articles
        ]

        if query.keywords:
            lowered_keywords = [value.lower() for value in query.keywords]
            records = [
                record
                for record in records
                if any(
                    keyword in f"{record.headline} {record.body or ''}".lower()
                    for keyword in lowered_keywords
                )
            ]

        records = [
            record
            for record in records
            if query.start_time <= record.timestamp <= query.end_time
        ]
        records.sort(key=lambda item: item.timestamp)
        return records[: query.limit]

    def subscribe_news(
        self,
        symbols: list[str],
        callback: Callable[[NewsRecord], None],
    ) -> int:
        """Start poll-based Benzinga subscription and return subscription id."""

        if not callable(callback):
            raise ValueError("callback must be callable")

        normalized_symbols = sorted(
            {value.strip().upper() for value in symbols if value.strip()}
        )
        if not normalized_symbols:
            raise ValueError("symbols must include at least one non-empty symbol")

        client = self._require_client()
        subscription_id = next(self._subscription_ids)
        stop_event = Event()

        def poll_loop() -> None:
            last_check = datetime.now(tz=UTC)
            while not stop_event.is_set():
                try:
                    self._rate_limiter.wait()
                    raw_articles = client.news(
                        company_tickers=",".join(normalized_symbols),
                        updated_since=last_check.strftime("%Y-%m-%dT%H:%M:%S"),
                        pagesize=50,
                        display_output="abstract",
                    )
                    records = [
                        self._parse_article(article, include_body=False)
                        for article in self._normalize_response(raw_articles)
                    ]
                    for record in sorted(records, key=lambda item: item.timestamp):
                        callback(record)
                    last_check = datetime.now(tz=UTC)
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.error("Benzinga polling error: %s", exc)

                stop_event.wait(self._poll_interval_seconds)

        thread = Thread(
            target=poll_loop,
            name=f"benzinga-news-{subscription_id}",
            daemon=True,
        )
        with self._lock:
            self._subscriptions[subscription_id] = (stop_event, thread)
        thread.start()
        return subscription_id

    def unsubscribe_news(self, subscription_id: int) -> None:
        """Stop an active Benzinga poll subscription."""

        with self._lock:
            subscription = self._subscriptions.pop(subscription_id, None)

        if subscription is None:
            return

        stop_event, thread = subscription
        stop_event.set()
        thread.join(timeout=max(self._poll_interval_seconds, 1.0))

    def get_available_categories(self) -> list[str]:
        """Return configured channels when provided by source config."""

        channels = self._config.get("channels")
        if isinstance(channels, list):
            return [str(value) for value in channels]
        return []

    def _require_client(self) -> Any:
        """Return connected provider client or raise connection error."""

        with self._lock:
            if not self._connected or self._client is None:
                raise DataSourceConnectionError("BenzingaNewsSource is not connected")
            return self._client

    def _normalize_response(self, response: object) -> list[dict[str, object]]:
        """Normalize Benzinga response shape to a list of article dictionaries."""

        if isinstance(response, list):
            return [item for item in response if isinstance(item, dict)]

        if isinstance(response, dict):
            data = response.get("data")
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]

        return []

    def _parse_article(
        self, article: dict[str, object], include_body: bool
    ) -> NewsRecord:
        """Map one Benzinga article payload to ``NewsRecord``."""

        created_raw = str(article.get("created", "")).strip()
        if not created_raw:
            raise DataSourceError("Benzinga article missing 'created' field")

        headline = str(article.get("title", "")).strip()
        if not headline:
            raise DataSourceError("Benzinga article missing 'title' field")

        body: str | None
        if include_body:
            body = self._as_optional_text(article.get("body"))
            if body is None:
                body = self._as_optional_text(article.get("teaser"))
        else:
            body = self._as_optional_text(article.get("teaser"))

        stocks = article.get("stocks", [])
        channels = article.get("channels", [])
        symbols = self._extract_values(stocks, ("name", "symbol", "ticker"))
        categories = self._extract_values(channels, ("name", "id", "slug"))

        url = self._as_optional_text(article.get("url"))
        news_id_raw = self._as_optional_text(article.get("id"))
        news_id = news_id_raw or generate_stable_id(url or "", created_raw)

        return NewsRecord(
            timestamp=parse_benzinga_datetime(created_raw),
            headline=headline,
            body=body,
            source="Benzinga",
            symbols=symbols,
            categories=categories,
            sentiment_score=None,
            url=url,
            news_id=news_id,
        )

    def _extract_values(
        self,
        payload: object,
        keys: tuple[str, ...],
    ) -> list[str]:
        """Extract string values from list payload items using candidate keys."""

        if not isinstance(payload, list):
            return []

        values: list[str] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            for key in keys:
                raw = item.get(key)
                if raw is None:
                    continue
                text = str(raw).strip()
                if text:
                    values.append(text)
                    break
        return values

    def _as_optional_text(self, value: object) -> str | None:
        """Return stripped text value or ``None`` for empty-like inputs."""

        if value is None:
            return None
        text = str(value).strip()
        return text or None


__all__ = ["BenzingaNewsSource"]
