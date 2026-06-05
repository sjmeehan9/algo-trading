"""API service for historical data acquisition workflows."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from algotrading.api.config import APIConfig
from algotrading.api.schemas.data_sources import (
    MarketDataProvider,
    NewsDataProvider,
    TrainingDataRequest,
    normalize_frequency,
    normalize_market_provider,
    normalize_training_data_request,
)
from algotrading.src.broker import BrokerRegistry
from algotrading.src.data_pipeline.acquisition import (
    AcquisitionResult,
    HistoricalMarketDataAcquirer,
    HistoricalNewsAcquirer,
)
from algotrading.src.data_pipeline.sources import NewsSourceFactory
from algotrading.src.data_pipeline.storage import DataManifest, LocalDataStore
from algotrading.src.models.registry.supporting_model_registry import (
    SupportingModelRegistry,
)
from fastapi import Request


class DataAcquisitionServiceError(Exception):
    """Base exception for data acquisition service operations."""


class DataAcquisitionService:
    """Coordinate canonical local storage with historical data acquirers."""

    def __init__(
        self,
        *,
        store: LocalDataStore,
        api_config: APIConfig,
        broker_registry: BrokerRegistry | None = None,
        news_source_factory: NewsSourceFactory | None = None,
        supporting_registry: SupportingModelRegistry | None = None,
    ) -> None:
        """Initialize the service with runtime configuration and storage."""

        self._store = store
        self._api_config = api_config
        self._broker_registry = broker_registry or BrokerRegistry()
        self._news_source_factory = news_source_factory
        self._supporting_registry = supporting_registry

    @property
    def store(self) -> LocalDataStore:
        """Return the canonical local data store used by this service."""

        return self._store

    def ensure_market_data(
        self,
        request: TrainingDataRequest | Mapping[str, object],
        data_config: Mapping[str, object] | None = None,
    ) -> AcquisitionResult:
        """Ensure historical market bars exist for a canonical or loose request."""

        canonical_request = _coerce_request(request, data_config)
        broker_name = _broker_settings_name(canonical_request.market_source.provider)
        acquirer = HistoricalMarketDataAcquirer(
            store=self._store,
            broker_registry=self._broker_registry,
            broker_config=self._api_config.get_broker_config(broker_name),
            broker_connection_params=self._api_config.get_broker_connection_params(
                broker_name
            ),
        )
        return acquirer.acquire(canonical_request)

    def ensure_news_data(
        self,
        request: TrainingDataRequest | Mapping[str, object],
        data_config: Mapping[str, object] | None = None,
        *,
        model: object | Mapping[str, object] | None = None,
    ) -> AcquisitionResult:
        """Ensure historical news exists when requested or required by a model."""

        canonical_request = _coerce_request(request, data_config)
        acquirer = HistoricalNewsAcquirer(
            store=self._store,
            news_source_factory=(
                self._news_source_factory
                or _build_news_source_factory(self._api_config)
            ),
            supporting_registry=self._supporting_registry,
            default_provider=self._api_config.news_provider,
            default_fallback_provider=self._api_config.news_fallback_provider,
        )
        return acquirer.acquire(canonical_request, model=model)

    def get_manifest(
        self,
        *,
        provider: object,
        symbol: str,
        frequency: object,
    ) -> DataManifest:
        """Read one canonical market-data manifest by provider, symbol, and frequency."""

        normalized_provider = normalize_market_provider(provider).value
        normalized_symbol = symbol.strip().upper()
        if not normalized_symbol:
            raise DataAcquisitionServiceError("symbol must be non-empty")
        normalized_frequency = normalize_frequency(frequency).value
        manifest_path = (
            self._store.root
            / "market"
            / normalized_provider
            / normalized_symbol
            / normalized_frequency
            / self._store.MANIFEST_FILENAME
        )
        return self._store.read_manifest(manifest_path)

    def get_news_manifest(
        self,
        *,
        provider: object,
        symbol: str,
    ) -> DataManifest:
        """Read one canonical news-data manifest by provider and symbol."""

        normalized_provider = _normalize_news_provider_value(provider)
        normalized_symbol = symbol.strip().upper()
        if not normalized_symbol:
            raise DataAcquisitionServiceError("symbol must be non-empty")
        manifest_path = (
            self._store.root
            / "news"
            / normalized_provider
            / normalized_symbol
            / self._store.MANIFEST_FILENAME
        )
        return self._store.read_manifest(manifest_path)


def create_default_data_acquisition_service(
    *,
    api_config: APIConfig,
    project_root: Path | None = None,
    store: LocalDataStore | None = None,
    broker_registry: BrokerRegistry | None = None,
) -> DataAcquisitionService:
    """Build a filesystem-backed data acquisition service."""

    root = project_root or Path(__file__).resolve().parents[4]
    local_store = store or LocalDataStore(root=root / "data" / "sourced")
    return DataAcquisitionService(
        store=local_store,
        api_config=api_config,
        broker_registry=broker_registry,
    )


def get_data_acquisition_service(request: Request) -> DataAcquisitionService:
    """FastAPI dependency resolver for the shared acquisition service."""

    service = getattr(request.app.state, "data_acquisition_service", None)
    if service is None:
        model_service = getattr(request.app.state, "model_service", None)
        supporting_registry = getattr(model_service, "supporting_registry", None)
        supporting_registry = getattr(
            request.app.state,
            "supporting_registry",
            supporting_registry,
        )
        service = create_default_data_acquisition_service(
            api_config=request.app.state.api_config,
            broker_registry=getattr(request.app.state, "broker_registry", None),
        )
        service._supporting_registry = supporting_registry
        request.app.state.data_acquisition_service = service
    return service


def _coerce_request(
    request: TrainingDataRequest | Mapping[str, object],
    data_config: Mapping[str, object] | None,
) -> TrainingDataRequest:
    if isinstance(request, TrainingDataRequest):
        if data_config is not None:
            return normalize_training_data_request(request.model_dump(), data_config)
        return request
    return normalize_training_data_request(request, data_config)


def _broker_settings_name(provider: MarketDataProvider) -> str:
    if provider == MarketDataProvider.IB:
        return "interactive_brokers"
    if provider == MarketDataProvider.ALPACA:
        return "alpaca"
    return provider.value


def _build_news_source_factory(api_config: APIConfig) -> NewsSourceFactory:
    provider_config: dict[str, object] = {
        "primary_provider": api_config.news_provider,
        "fallback_provider": api_config.news_fallback_provider,
        "providers": {
            "benzinga": {
                "source_id": "benzinga_news",
                "calls_per_minute": 100,
            },
            "alphavantage": {
                "source_id": "alphavantage_news",
                "base_url": "https://www.alphavantage.co/query",
                "calls_per_minute": 5,
            },
        },
    }
    credentials_config: dict[str, object] = {
        "benzinga": {"api_key": api_config.benzinga_api_key or ""},
        "alphavantage": {"api_key": api_config.alphavantage_api_key or ""},
    }
    return NewsSourceFactory(
        provider_config=provider_config,
        credentials_config=credentials_config,
    )


def _normalize_news_provider_value(provider: object) -> str:
    if isinstance(provider, NewsDataProvider):
        return provider.value
    normalized = str(provider).strip().lower()
    if normalized in {"alpha_vantage", "alpha-vantage"}:
        return NewsDataProvider.ALPHAVANTAGE.value
    return normalized


__all__ = [
    "DataAcquisitionService",
    "DataAcquisitionServiceError",
    "create_default_data_acquisition_service",
    "get_data_acquisition_service",
]
