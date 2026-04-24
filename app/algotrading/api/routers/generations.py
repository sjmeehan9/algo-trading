"""Generation history and comparison endpoints."""

from __future__ import annotations

from algotrading.api.schemas.common import APIResponse
from algotrading.api.schemas.generations import (
    GenerationComparison,
    GenerationComparisonRequest,
    GenerationDetail,
)
from algotrading.api.services import (
    GenerationNotFoundError,
    ModelNotFoundError,
    ModelService,
    ModelValidationError,
    get_model_service,
)
from fastapi import APIRouter, Depends, HTTPException

router = APIRouter(prefix="/generations", tags=["generations"])


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


@router.get("/{generation_id}", response_model=APIResponse[GenerationDetail])
def get_generation(
    generation_id: str,
    service: ModelService = Depends(get_model_service),
) -> APIResponse[GenerationDetail]:
    """Get one generation by generation ID."""

    try:
        detail = service.get_generation_detail_by_id(generation_id)
    except GenerationNotFoundError as exc:
        raise _http_error(404, str(exc), "GENERATION_NOT_FOUND") from exc

    return APIResponse(data=detail)


@router.post("/compare", response_model=APIResponse[GenerationComparison])
def compare_generations(
    request: GenerationComparisonRequest,
    service: ModelService = Depends(get_model_service),
) -> APIResponse[GenerationComparison]:
    """Compare selected generations for a model."""

    try:
        comparison = service.compare_generations(
            model_id=request.model_id,
            generation_ids=request.generation_ids,
        )
    except ModelNotFoundError as exc:
        raise _http_error(404, str(exc), "MODEL_NOT_FOUND") from exc
    except GenerationNotFoundError as exc:
        raise _http_error(404, str(exc), "GENERATION_NOT_FOUND") from exc
    except ModelValidationError as exc:
        raise _http_error(400, str(exc), "MODEL_VALIDATION_ERROR") from exc

    return APIResponse(data=comparison)


__all__ = ["router"]
