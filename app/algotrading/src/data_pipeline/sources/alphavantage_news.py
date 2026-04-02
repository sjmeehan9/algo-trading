"""Alpha Vantage news provider implementation for fallback sentiment ingestion."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from itertools import count
from threading import Event, RLock, Thread

import requests
from algotrading.src.data_pipeline.sources.exceptions import (
    DataSourceConnectionError,
    DataSourceError,
)
from algotrading.src.data_pipeline.sources.news_source import NewsDataSource
from algotrading.src.data_pipeline.sources.news_utils import (
    RateLimiter,
    ensure_utc,
    generate_stable_id,
)
from algotrading.src.data_pipeline.types import DataType, NewsQuery, NewsRecord

logger = logging.getLogger(__name__)

_ALPHA_VANTAGE_TOPICS = [
    "blockchain",
    "earnings",
    "ipo",
    "mergers_and_acquisitions",
    "financial_markets",
    "economy_fiscal",
    "economy_monetary",
    "economy_macro",
    "energy_transportation",
    "finance",
    "life_sciences",
    "manufacturing",
    "real_estate",
    "retail_wholesale",
    "technology",
]


class AlphaVantageNewsSource(NewsDataSource):
    """Fallback news source using Alpha Vantage NEWS_SENTIMENT API.

    Alpha Vantage offers no push stream transport for news, so ``subscribe_news``
    is implemented with polling constrained by provider rate limits.
    """

    def __init__(self, api_key: str, config: dict[str, object] | None = None) -> None:
        """Initialize source with API key and optional provider configuration."""

        if not api_key.strip():
            raise ValueError("api_key must be non-empty")

        self._api_key = api_key.strip()
        self._config = dict(config or {})
        self._base_url = str(
            self._config.get("base_url", "https://www.alphavantage.co/query")
        )
        self._source_id = str(self._config.get("source_id", "alphavantage_news"))
        self._rate_limiter = RateLimiter(
            calls_per_minute=int(self._config.get("calls_per_minute", 5))
        )
        self._poll_interval_seconds = float(
            self._config.get("poll_interval_seconds", 60)
        )
        if self._poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be greater than 0")

        self._session = requests.Session()
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
                "provider": "alphavantage",
                "poll_interval_seconds": self._poll_interval_seconds,
                "calls_per_minute": 5,
                "provides_sentiment": True,
            }
        )
        return metadata

    def connect(self) -> None:
        """Verify API key with a minimal provider request."""

        params = {
            "function": "NEWS_SENTIMENT",
            "apikey": self._api_key,
            "limit": 1,
        }
        try:
            response = self._session.get(self._base_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # pragma: no cover - network/provider behavior
            raise DataSourceConnectionError(
                f"Alpha Vantage API connection failed: {exc}"
            ) from exc

        if "Error Message" in data:
            raise DataSourceConnectionError(str(data["Error Message"]))

        with self._lock:
            self._connected = True

    def disconnect(self) -> None:
        """Stop active subscriptions and close the HTTP session."""

        with self._lock:
            subscription_ids = list(self._subscriptions.keys())

        for subscription_id in subscription_ids:
            self.unsubscribe_news(subscription_id)

        with self._lock:
            self._connected = False

        self._session.close()

    def fetch_news(self, query: NewsQuery) -> list[NewsRecord]:
        """Fetch historical Alpha Vantage news mapped to ``NewsRecord``."""

        self._require_connected()
        self._rate_limiter.wait()

        params: dict[str, object] = {
            "function": "NEWS_SENTIMENT",
            "apikey": self._api_key,
            "tickers": ",".join(query.symbols) if query.symbols else None,
            "time_from": query.start_time.strftime("%Y%m%dT%H%M"),
            "time_to": query.end_time.strftime("%Y%m%dT%H%M"),
            "limit": min(query.limit, 1000),
            "sort": "LATEST",
        }
        if query.categories:
            params["topics"] = ",".join(query.categories)
        clean_params = {
            key: value for key, value in params.items() if value is not None
        }

        try:
            response = self._session.get(
                self._base_url, params=clean_params, timeout=20
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # pragma: no cover - network/provider behavior
            raise DataSourceError(f"Alpha Vantage news request failed: {exc}") from exc

        if "Error Message" in payload:
            raise DataSourceError(str(payload["Error Message"]))
        if "Note" in payload and not payload.get("feed"):
            raise DataSourceError(str(payload["Note"]))

        raw_feed = payload.get("feed", [])
        feed = [item for item in raw_feed if isinstance(item, dict)]
        records = [self._parse_article(item) for item in feed]

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
        """Start poll-based subscription constrained by provider rate limits."""

        if not callable(callback):
            raise ValueError("callback must be callable")

        normalized_symbols = sorted(
            {value.strip().upper() for value in symbols if value.strip()}
        )
        if not normalized_symbols:
            raise ValueError("symbols must include at least one non-empty symbol")

        self._require_connected()

        subscription_id = next(self._subscription_ids)
        stop_event = Event()

        def poll_loop() -> None:
            last_check = datetime.now(tz=UTC)
            while not stop_event.is_set():
                try:
                    query = NewsQuery(
                        start_time=last_check,
                        end_time=datetime.now(tz=UTC),
                        symbols=normalized_symbols,
                        limit=50,
                        include_body=False,
                    )
                    records = self.fetch_news(query)
                    for record in sorted(records, key=lambda item: item.timestamp):
                        callback(record)
                    last_check = datetime.now(tz=UTC)
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.error("Alpha Vantage polling error: %s", exc)

                stop_event.wait(self._poll_interval_seconds)

        thread = Thread(
            target=poll_loop,
            name=f"alphavantage-news-{subscription_id}",
            daemon=True,
        )

        with self._lock:
            self._subscriptions[subscription_id] = (stop_event, thread)
        thread.start()
        return subscription_id

    def unsubscribe_news(self, subscription_id: int) -> None:
        """Stop an active Alpha Vantage poll subscription."""

        with self._lock:
            subscription = self._subscriptions.pop(subscription_id, None)

        if subscription is None:
            return

        stop_event, thread = subscription
        stop_event.set()
        thread.join(timeout=max(self._poll_interval_seconds, 1.0))

    def get_available_categories(self) -> list[str]:
        """Return supported Alpha Vantage topic filter values."""

        return list(_ALPHA_VANTAGE_TOPICS)

    def _parse_article(self, item: dict[str, object]) -> NewsRecord:
        """Map one Alpha Vantage feed item to ``NewsRecord``."""

        published_raw = str(item.get("time_published", "")).strip()
        if not published_raw:
            raise DataSourceError("Alpha Vantage article missing 'time_published'")

        headline = str(item.get("title", "")).strip()
        if not headline:
            raise DataSourceError("Alpha Vantage article missing 'title'")

        try:
            timestamp = ensure_utc(datetime.strptime(published_raw, "%Y%m%dT%H%M%S"))
        except ValueError as exc:
            raise DataSourceError(
                f"Invalid Alpha Vantage time_published value: {published_raw}"
            ) from exc

        ticker_sentiment = item.get("ticker_sentiment", [])
        symbols = []
        if isinstance(ticker_sentiment, list):
            for value in ticker_sentiment:
                if isinstance(value, dict) and value.get("ticker") is not None:
                    ticker = str(value["ticker"]).strip()
                    if ticker:
                        symbols.append(ticker)

        topics = item.get("topics", [])
        categories = []
        if isinstance(topics, list):
            for value in topics:
                if isinstance(value, dict) and value.get("topic") is not None:
                    topic = str(value["topic"]).strip()
                    if topic:
                        categories.append(topic)

        sentiment_score = None
        score_raw = item.get("overall_sentiment_score")
        if score_raw is not None:
            try:
                sentiment_score = float(score_raw)
            except (TypeError, ValueError):
                sentiment_score = None

        url = self._as_optional_text(item.get("url"))
        news_id = self._as_optional_text(item.get("news_id")) or generate_stable_id(
            url or "", published_raw
        )

        return NewsRecord(
            timestamp=timestamp,
            headline=headline,
            body=self._as_optional_text(item.get("summary")),
            source=str(item.get("source", "Alpha Vantage")).strip() or "Alpha Vantage",
            symbols=symbols,
            categories=categories,
            sentiment_score=sentiment_score,
            url=url,
            news_id=news_id,
        )

    def _require_connected(self) -> None:
        """Raise when source is not connected."""

        if not self._connected:
            raise DataSourceConnectionError("AlphaVantageNewsSource is not connected")

    def _as_optional_text(self, value: object) -> str | None:
        """Return stripped text value or ``None`` for empty-like inputs."""

        if value is None:
            return None
        text = str(value).strip()
        return text or None


__all__ = ["AlphaVantageNewsSource"]
