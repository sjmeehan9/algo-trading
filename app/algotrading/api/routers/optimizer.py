"""REST endpoints for LLM hyperparameter optimization."""

from __future__ import annotations

from algotrading.api.schemas.common import APIResponse
from algotrading.api.schemas.optimizer import (
    AppliedSuggestionResponse,
    ApplySuggestionRequest,
    OptimizationResult,
    OptimizerAnalyzeRequest,
)
from algotrading.api.services import ModelNotFoundError
from algotrading.api.services.llm_optimizer import (
    LLMOptimizer,
    OptimizerServiceError,
    OptimizerSuggestionNotFoundError,
    OptimizerValidationError,
    get_optimizer_service,
)
from fastapi import APIRouter, Depends, HTTPException, status

router = APIRouter(prefix="/optimizer", tags=["optimizer"])


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


@router.post(
    "/analyze",
    response_model=APIResponse[OptimizationResult],
    status_code=status.HTTP_201_CREATED,
)
def analyze_hyperparameters(
    request: OptimizerAnalyzeRequest,
    service: LLMOptimizer = Depends(get_optimizer_service),
) -> APIResponse[OptimizationResult]:
    """Analyze model history and create hyperparameter suggestions."""

    try:
        result = service.analyze(request)
    except ModelNotFoundError as exc:
        raise _http_error(404, str(exc), "MODEL_NOT_FOUND") from exc
    except OptimizerValidationError as exc:
        raise _http_error(400, str(exc), "OPTIMIZER_VALIDATION_ERROR") from exc
    except OptimizerServiceError as exc:
        raise _http_error(500, str(exc), "OPTIMIZER_SERVICE_ERROR") from exc

    return APIResponse(data=result, message="Hyperparameter analysis completed")


@router.get(
    "/suggestions/{model_id}",
    response_model=APIResponse[OptimizationResult | None],
)
def get_latest_suggestions(
    model_id: str,
    service: LLMOptimizer = Depends(get_optimizer_service),
) -> APIResponse[OptimizationResult | None]:
    """Return the latest optimizer result for a model if available."""

    try:
        result = service.get_latest_result(model_id)
    except ModelNotFoundError as exc:
        raise _http_error(404, str(exc), "MODEL_NOT_FOUND") from exc
    return APIResponse(data=result)


@router.post("/apply", response_model=APIResponse[AppliedSuggestionResponse])
def apply_suggestion(
    request: ApplySuggestionRequest,
    service: LLMOptimizer = Depends(get_optimizer_service),
) -> APIResponse[AppliedSuggestionResponse]:
    """Apply one suggestion to the model configuration."""

    try:
        response = service.apply_suggestion(request)
    except ModelNotFoundError as exc:
        raise _http_error(404, str(exc), "MODEL_NOT_FOUND") from exc
    except OptimizerSuggestionNotFoundError as exc:
        raise _http_error(404, str(exc), "OPTIMIZER_SUGGESTION_NOT_FOUND") from exc
    except OptimizerValidationError as exc:
        raise _http_error(409, str(exc), "OPTIMIZER_VALIDATION_ERROR") from exc
    except OptimizerServiceError as exc:
        raise _http_error(500, str(exc), "OPTIMIZER_SERVICE_ERROR") from exc

    return APIResponse(data=response, message="Suggestion applied")


__all__ = ["router"]
