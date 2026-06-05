"""REST endpoints for historical data acquisition."""

from __future__ import annotations

from collections.abc import Mapping

from algotrading.api.schemas.common import APIResponse
from algotrading.api.schemas.data_sources import (
    DataSourceConfigError,
    normalize_training_data_request,
)
from algotrading.api.services.data_acquisition_service import (
    DataAcquisitionService,
    DataAcquisitionServiceError,
    get_data_acquisition_service,
)
from algotrading.src.data_pipeline.acquisition import (
    AcquisitionResult,
    MarketDataAcquisitionError,
    NewsDataAcquisitionError,
)
from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import ValidationError

router = APIRouter(prefix="/data", tags=["data"])


def _http_error(
    status_code: int,
    error: str,
    error_code: str,
    details: dict[str, object] | None = None,
) -> HTTPException:
    """Build a standardized HTTPException payload for data endpoints."""

    return HTTPException(
        status_code=status_code,
        detail={
            "error": error,
            "error_code": error_code,
            "details": details,
        },
    )


@router.post("/market/acquire", response_model=APIResponse[AcquisitionResult])
def acquire_market_data(
    payload: dict[str, object] = Body(...),
    service: DataAcquisitionService = Depends(get_data_acquisition_service),
) -> APIResponse[AcquisitionResult]:
    """Acquire historical market data and persist it to the canonical store."""

    try:
        request = normalize_training_data_request(payload)
        result = service.ensure_market_data(request)
    except (DataSourceConfigError, ValidationError) as exc:
        raise _http_error(422, str(exc), "DATA_REQUEST_VALIDATION_ERROR") from exc
    except MarketDataAcquisitionError as exc:
        raise _http_error(
            400,
            str(exc),
            "MARKET_DATA_ACQUISITION_ERROR",
        ) from exc
    except DataAcquisitionServiceError as exc:
        raise _http_error(400, str(exc), "DATA_ACQUISITION_ERROR") from exc

    return APIResponse(data=result, message="Market data acquisition complete")


@router.post("/news/acquire", response_model=APIResponse[AcquisitionResult])
def acquire_news_data(
    payload: dict[str, object] = Body(...),
    service: DataAcquisitionService = Depends(get_data_acquisition_service),
) -> APIResponse[AcquisitionResult]:
    """Acquire historical news data and persist it to the canonical store."""

    try:
        request_payload = payload.get("data_config", payload)
        if not isinstance(request_payload, Mapping):
            raise DataSourceConfigError("data_config must be an object when provided")
        model_payload = payload.get("model")
        model = model_payload if isinstance(model_payload, Mapping) else None
        request = normalize_training_data_request(request_payload)
        result = service.ensure_news_data(request, model=model)
    except (DataSourceConfigError, ValidationError) as exc:
        raise _http_error(422, str(exc), "DATA_REQUEST_VALIDATION_ERROR") from exc
    except NewsDataAcquisitionError as exc:
        raise _http_error(
            400,
            str(exc),
            "NEWS_DATA_ACQUISITION_ERROR",
        ) from exc
    except DataAcquisitionServiceError as exc:
        raise _http_error(400, str(exc), "DATA_ACQUISITION_ERROR") from exc

    return APIResponse(data=result, message="News data acquisition complete")


__all__ = ["router"]
