"""Historical news acquisition backed by cache and configured providers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from algotrading.api.schemas.data_sources import (
    CachePolicy,
    NewsDataProvider,
    TrainingDataRequest,
    normalize_news_provider,
)
from algotrading.src.data_pipeline import DataFrequency, DataType, NewsQuery
from algotrading.src.data_pipeline.acquisition.market_data import (
    AcquisitionResult,
    AcquisitionStatus,
    SymbolAcquisitionReport,
)
from algotrading.src.data_pipeline.sources import (
    MockNewsSource,
    NewsDataSource,
    NewsGenerator,
    NewsSourceFactory,
)
from algotrading.src.data_pipeline.sources.exceptions import DataSourceError
from algotrading.src.data_pipeline.storage import (
    DataManifest,
    LocalDataStore,
    LocalDataStoreError,
)
from algotrading.src.models.registry.supporting_model_registry import (
    SupportingModelRegistry,
)


class NewsDataAcquisitionError(DataSourceError):
    """Raised when historical news cannot be acquired or validated."""

    def __init__(
        self,
        message: str,
        reports: Sequence[SymbolAcquisitionReport] | None = None,
    ) -> None:
        """Initialize a news acquisition error with optional partial reports."""

        super().__init__(message)
        self.reports = list(reports or [])


class HistoricalNewsAcquirer:
    """Ensure historical news records exist in the canonical local store."""

    def __init__(
        self,
        *,
        store: LocalDataStore | None = None,
        news_source_factory: NewsSourceFactory | None = None,
        supporting_registry: SupportingModelRegistry | None = None,
        default_provider: object = NewsDataProvider.BENZINGA,
        default_fallback_provider: object | None = NewsDataProvider.ALPHAVANTAGE,
    ) -> None:
        """Initialize the acquirer with storage and provider collaborators."""

        self._store = store or LocalDataStore()
        self._news_source_factory = news_source_factory
        self._supporting_registry = supporting_registry
        self._default_provider = normalize_news_provider(default_provider)
        self._default_fallback_provider = (
            normalize_news_provider(default_fallback_provider)
            if default_fallback_provider is not None
            else None
        )

    @property
    def store(self) -> LocalDataStore:
        """Return the canonical local data store used by this acquirer."""

        return self._store

    def acquire(
        self,
        request: TrainingDataRequest,
        model: object | Mapping[str, object] | None = None,
    ) -> AcquisitionResult:
        """Acquire or load historical news for a canonical data request."""

        required_by_model = model_requires_news(model, self._supporting_registry)
        effective_request = self._with_effective_news_source(
            request,
            required_by_model=required_by_model,
        )
        required = effective_request.news_source.required or required_by_model
        warnings: list[str] = []

        if not effective_request.news_source.enabled:
            return _result_from_reports(
                effective_request,
                reports=[],
                warnings=[],
                provider=NewsDataProvider.NONE.value,
            )

        if effective_request.cache_policy != CachePolicy.REFRESH:
            cached = self._try_cache(effective_request, required, warnings)
            if cached is not None:
                return cached

        if effective_request.cache_policy == CachePolicy.REQUIRE_CACHE:
            return _result_from_reports(
                effective_request,
                reports=[
                    _optional_unavailable_report(
                        symbol=symbol,
                        provider=_provider_value(
                            effective_request.news_source.provider
                        ),
                        message="Required cache policy prevented provider fetch.",
                    )
                    for symbol in effective_request.symbols
                ],
                warnings=warnings,
            )

        if effective_request.news_source.provider == NewsDataProvider.NONE:
            message = "Historical news is required but no news provider is configured."
            if required:
                raise NewsDataAcquisitionError(message)
            warnings.append("News acquisition skipped because provider is 'none'.")
            return _result_from_reports(
                effective_request, reports=[], warnings=warnings
            )

        reports = [
            self._acquire_symbol(effective_request, symbol=symbol, required=required)
            for symbol in effective_request.symbols
        ]
        return _result_from_reports(
            effective_request, reports=reports, warnings=warnings
        )

    def _with_effective_news_source(
        self,
        request: TrainingDataRequest,
        *,
        required_by_model: bool,
    ) -> TrainingDataRequest:
        news_source = request.news_source
        updates: dict[str, object] = {}

        if required_by_model:
            updates["enabled"] = True
            updates["required"] = True
            if news_source.provider == NewsDataProvider.NONE:
                updates["provider"] = self._default_provider

        if news_source.provider != NewsDataProvider.NONE and news_source.enabled:
            if news_source.fallback_provider is None:
                updates["fallback_provider"] = self._default_fallback_provider

        if not updates:
            return request

        updated_news_source = news_source.model_copy(update=updates)
        return request.model_copy(update={"news_source": updated_news_source})

    def _try_cache(
        self,
        request: TrainingDataRequest,
        required: bool,
        warnings: list[str],
    ) -> AcquisitionResult | None:
        cache_errors: list[str] = []
        for provider in _candidate_providers(request):
            cached_request = _request_for_provider(request, provider)
            try:
                cached = self._store.find_news_data(cached_request)
            except LocalDataStoreError as exc:
                cache_errors.append(f"{_provider_value(provider)}: {exc}")
                continue

            reports = [
                _report_from_manifest(
                    symbol=symbol,
                    manifest=stored.manifest,
                    status=AcquisitionStatus.CACHED,
                    row_count=len(stored.records),
                    data_path=stored.data_path,
                    manifest_path=stored.manifest_path,
                )
                for symbol, stored in sorted(cached.items())
            ]
            return _result_from_reports(
                cached_request,
                reports=reports,
                warnings=[],
                provider=_provider_value(provider),
            )

        message = "News cache miss: " + "; ".join(cache_errors)
        if request.cache_policy == CachePolicy.REQUIRE_CACHE and required:
            raise NewsDataAcquisitionError(
                f"Required news cache is unavailable: {'; '.join(cache_errors)}"
            )
        warnings.append(message)
        return None

    def _acquire_symbol(
        self,
        request: TrainingDataRequest,
        *,
        symbol: str,
        required: bool,
    ) -> SymbolAcquisitionReport:
        attempts: list[str] = []
        for provider in _candidate_providers(request):
            try:
                records = self._fetch_provider_records(provider, request, symbol)
                manifest = self._store.write_news_records(
                    records,
                    provider=provider,
                    symbol=symbol,
                    requested_start=request.start_time,
                    requested_end=request.end_time,
                    cache_policy=request.cache_policy,
                )
            except Exception as exc:
                attempts.append(f"{_provider_value(provider)}: {exc}")
                continue

            return _report_from_manifest(
                symbol=symbol,
                manifest=manifest,
                status=AcquisitionStatus.ACQUIRED,
                row_count=manifest.record_count,
            )

        message = (
            f"No historical news records were available for symbol '{symbol}' "
            f"between {request.start_time.isoformat()} and "
            f"{request.end_time.isoformat()}. Attempts: {'; '.join(attempts)}"
        )
        if required:
            raise NewsDataAcquisitionError(message)
        return _optional_unavailable_report(
            symbol=symbol,
            provider=_provider_value(request.news_source.provider),
            message=message,
        )

    def _fetch_provider_records(
        self,
        provider: NewsDataProvider,
        request: TrainingDataRequest,
        symbol: str,
    ) -> list[object]:
        source = self._create_source(provider, request)
        query = NewsQuery(
            start_time=request.start_time,
            end_time=request.end_time,
            symbols=[symbol],
            keywords=list(request.news_source.keywords),
            categories=list(request.news_source.categories),
            limit=request.news_source.limit,
            include_body=request.news_source.include_body,
        )

        try:
            source.connect()
            return list(source.fetch_news(query))
        finally:
            source.disconnect()

    def _create_source(
        self,
        provider: NewsDataProvider,
        request: TrainingDataRequest,
    ) -> NewsDataSource:
        if provider == NewsDataProvider.MOCK:
            return MockNewsSource(
                symbols=list(request.symbols),
                generator=NewsGenerator(seed=17),
            )
        if self._news_source_factory is None:
            raise NewsDataAcquisitionError(
                "A NewsSourceFactory is required for real news providers."
            )
        return self._news_source_factory.create(provider.value)


def model_requires_news(
    model: object | Mapping[str, object] | None,
    supporting_registry: SupportingModelRegistry | None,
) -> bool:
    """Return true when a model references NEWS_TEXT supporting inputs."""

    if model is None:
        return False

    supporting_model_ids = _supporting_model_ids(model)
    if not supporting_model_ids:
        return False
    if supporting_registry is None:
        raise NewsDataAcquisitionError(
            "Cannot inspect supporting model inputs without a supporting registry."
        )

    for model_id in supporting_model_ids:
        entry = supporting_registry.get(model_id)
        if entry is None:
            raise NewsDataAcquisitionError(
                f"Supporting model '{model_id}' referenced by model data request "
                "was not found in the supporting model registry."
            )
        if any(_is_news_text(value) for value in entry.config.input_data_types):
            return True
    return False


def _supporting_model_ids(model: object | Mapping[str, object]) -> list[str]:
    if isinstance(model, Mapping):
        raw_values = model.get("supporting_model_ids", [])
    else:
        raw_values = getattr(model, "supporting_model_ids", [])
    if raw_values is None:
        return []
    if isinstance(raw_values, str):
        return [raw_values.strip()] if raw_values.strip() else []
    return [str(value).strip() for value in list(raw_values) if str(value).strip()]


def _is_news_text(value: object) -> bool:
    if value == DataType.NEWS_TEXT:
        return True
    normalized = str(value).strip()
    return normalized in {DataType.NEWS_TEXT.value, "NEWS_TEXT", "news_text"}


def _candidate_providers(request: TrainingDataRequest) -> list[NewsDataProvider]:
    providers = [request.news_source.provider]
    fallback = request.news_source.fallback_provider
    if fallback is not None and fallback not in providers:
        providers.append(fallback)
    return [provider for provider in providers if provider != NewsDataProvider.NONE]


def _request_for_provider(
    request: TrainingDataRequest,
    provider: NewsDataProvider,
) -> TrainingDataRequest:
    news_source = request.news_source.model_copy(update={"provider": provider})
    return request.model_copy(update={"news_source": news_source})


def _result_from_reports(
    request: TrainingDataRequest,
    *,
    reports: list[SymbolAcquisitionReport],
    warnings: list[str],
    provider: str | None = None,
) -> AcquisitionResult:
    return AcquisitionResult(
        provider=provider or _provider_value(request.news_source.provider),
        cache_policy=request.cache_policy.value,
        frequency=DataFrequency.IRREGULAR,
        requested_start=request.start_time,
        requested_end=request.end_time,
        reports=reports,
        warnings=warnings,
    )


def _report_from_manifest(
    *,
    symbol: str,
    manifest: DataManifest,
    status: AcquisitionStatus,
    row_count: int,
    data_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> SymbolAcquisitionReport:
    data_paths = list(manifest.data_file_paths)
    resolved_data_path = str(data_path or (data_paths[0] if data_paths else "")) or None
    resolved_manifest_path = str(manifest_path or manifest.manifest_path or "") or None
    return SymbolAcquisitionReport(
        symbol=symbol,
        status=status,
        provider=manifest.provider,
        row_count=row_count,
        data_path=resolved_data_path,
        manifest_path=resolved_manifest_path,
        actual_start=manifest.actual_start,
        actual_end=manifest.actual_end,
        source_file_paths=list(manifest.source_file_paths),
    )


def _optional_unavailable_report(
    *,
    symbol: str,
    provider: str,
    message: str,
) -> SymbolAcquisitionReport:
    return SymbolAcquisitionReport(
        symbol=symbol,
        status=AcquisitionStatus.OPTIONAL_UNAVAILABLE,
        provider=provider,
        row_count=0,
        warnings=[message],
    )


def _provider_value(provider: object) -> str:
    if isinstance(provider, NewsDataProvider):
        return provider.value
    return str(provider).strip().lower()


__all__ = [
    "HistoricalNewsAcquirer",
    "NewsDataAcquisitionError",
    "model_requires_news",
]
