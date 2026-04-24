"""Model management REST endpoints."""

from __future__ import annotations

from algotrading.api.schemas.common import APIResponse, PaginatedResponse
from algotrading.api.schemas.generations import GenerationSummary
from algotrading.api.schemas.models import (
    ModelConfigCreate,
    ModelConfigResponse,
    ModelConfigUpdate,
    ModelType,
)
from algotrading.api.services import (
    InvalidModelStateError,
    ModelNotFoundError,
    ModelService,
    ModelValidationError,
    get_model_service,
)
from fastapi import APIRouter, Depends, HTTPException, Query, status

router = APIRouter(prefix="/models", tags=["models"])


def _http_error(
    status_code: int,
    error: str,
    error_code: str,
    details: dict[str, object] | None = None,
) -> HTTPException:
    """Build a consistent HTTPException payload consumed by global handlers."""

    return HTTPException(
        status_code=status_code,
        detail={
            "error": error,
            "error_code": error_code,
            "details": details,
        },
    )


@router.get("", response_model=PaginatedResponse[ModelConfigResponse])
def list_models(
    model_type: list[ModelType] | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    service: ModelService = Depends(get_model_service),
) -> PaginatedResponse[ModelConfigResponse]:
    """List all model configurations with optional model-type filters."""

    try:
        return service.list_models(
            model_types=model_type, page=page, page_size=page_size
        )
    except ModelValidationError as exc:
        raise _http_error(400, str(exc), "MODEL_VALIDATION_ERROR") from exc


@router.post(
    "",
    response_model=APIResponse[ModelConfigResponse],
    status_code=status.HTTP_201_CREATED,
)
def create_model(
    config: ModelConfigCreate,
    service: ModelService = Depends(get_model_service),
) -> APIResponse[ModelConfigResponse]:
    """Create a model configuration."""

    try:
        created = service.create_model(config)
    except ModelValidationError as exc:
        raise _http_error(400, str(exc), "MODEL_VALIDATION_ERROR") from exc
    except InvalidModelStateError as exc:
        raise _http_error(409, str(exc), "MODEL_STATE_ERROR") from exc

    return APIResponse(data=created, message="Model created successfully")


@router.get("/{model_id}", response_model=APIResponse[ModelConfigResponse])
def get_model(
    model_id: str,
    service: ModelService = Depends(get_model_service),
) -> APIResponse[ModelConfigResponse]:
    """Get one model configuration by ID."""

    try:
        model = service.get_model(model_id)
    except ModelNotFoundError as exc:
        raise _http_error(404, str(exc), "MODEL_NOT_FOUND") from exc

    return APIResponse(data=model)


@router.put("/{model_id}", response_model=APIResponse[ModelConfigResponse])
def update_model(
    model_id: str,
    updates: ModelConfigUpdate,
    service: ModelService = Depends(get_model_service),
) -> APIResponse[ModelConfigResponse]:
    """Update an existing model configuration."""

    try:
        updated = service.update_model(model_id, updates)
    except ModelNotFoundError as exc:
        raise _http_error(404, str(exc), "MODEL_NOT_FOUND") from exc
    except ModelValidationError as exc:
        raise _http_error(400, str(exc), "MODEL_VALIDATION_ERROR") from exc
    except InvalidModelStateError as exc:
        raise _http_error(409, str(exc), "MODEL_STATE_ERROR") from exc

    return APIResponse(data=updated, message="Model updated successfully")


@router.delete("/{model_id}", response_model=APIResponse[dict[str, str]])
def delete_model(
    model_id: str,
    service: ModelService = Depends(get_model_service),
) -> APIResponse[dict[str, str]]:
    """Delete one model configuration."""

    try:
        service.delete_model(model_id)
    except ModelNotFoundError as exc:
        raise _http_error(404, str(exc), "MODEL_NOT_FOUND") from exc

    return APIResponse(
        data={"model_id": model_id},
        message="Model deleted successfully",
    )


@router.get(
    "/{model_id}/generations",
    response_model=PaginatedResponse[GenerationSummary],
)
def list_generations(
    model_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    service: ModelService = Depends(get_model_service),
) -> PaginatedResponse[GenerationSummary]:
    """List generation history for one model."""

    try:
        return service.get_generations(
            model_id=model_id, page=page, page_size=page_size
        )
    except ModelNotFoundError as exc:
        raise _http_error(404, str(exc), "MODEL_NOT_FOUND") from exc
    except ModelValidationError as exc:
        raise _http_error(400, str(exc), "MODEL_VALIDATION_ERROR") from exc


__all__ = ["router"]
