"""REST endpoints for pre-deployment model generation selection."""

from __future__ import annotations

from algotrading.api.schemas.common import APIResponse
from algotrading.api.schemas.deployment import (
    DeployableModel,
    DeploymentCandidate,
    DeploymentReadiness,
    DeploymentSelection,
    DeploymentValidationRequest,
)
from algotrading.api.services import GenerationNotFoundError, ModelNotFoundError
from algotrading.api.services.deployment_service import (
    DeploymentSelectionNotFoundError,
    DeploymentService,
    DeploymentServiceError,
    DeploymentValidationError,
    get_deployment_service,
)
from fastapi import APIRouter, Depends, HTTPException, Request, status

router = APIRouter(prefix="/deployment", tags=["deployment"])


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


@router.get("/candidates", response_model=APIResponse[list[DeploymentCandidate]])
def list_candidates(
    service: DeploymentService = Depends(get_deployment_service),
) -> APIResponse[list[DeploymentCandidate]]:
    """List core RL candidates available for pre-deployment selection."""

    candidates = service.list_candidates()
    return APIResponse(data=candidates)


@router.get("/deployable-models", response_model=APIResponse[list[DeployableModel]])
def list_deployable_models(
    service: DeploymentService = Depends(get_deployment_service),
) -> APIResponse[list[DeployableModel]]:
    """List trained core RL models allowed through the hard deployment gate."""

    return APIResponse(data=service.list_deployable_models())


@router.post("/validate", response_model=APIResponse[DeploymentReadiness])
def validate_candidate(
    request: DeploymentValidationRequest,
    service: DeploymentService = Depends(get_deployment_service),
) -> APIResponse[DeploymentReadiness]:
    """Validate one model/generation pair for deployment readiness."""

    try:
        readiness = service.validate_selection(request)
    except ModelNotFoundError as exc:
        raise _http_error(404, str(exc), "MODEL_NOT_FOUND") from exc
    except GenerationNotFoundError as exc:
        raise _http_error(404, str(exc), "GENERATION_NOT_FOUND") from exc
    except DeploymentValidationError as exc:
        raise _http_error(400, str(exc), "DEPLOYMENT_VALIDATION_ERROR") from exc
    return APIResponse(data=readiness)


@router.get("/selection", response_model=APIResponse[DeploymentSelection | None])
def get_selection(
    service: DeploymentService = Depends(get_deployment_service),
) -> APIResponse[DeploymentSelection | None]:
    """Return the persisted deployment selection, if one exists."""

    try:
        selection = service.get_selection()
    except DeploymentSelectionNotFoundError:
        return APIResponse(data=None)
    except (ModelNotFoundError, GenerationNotFoundError) as exc:
        raise _http_error(409, str(exc), "DEPLOYMENT_SELECTION_STALE") from exc
    except DeploymentValidationError as exc:
        raise _http_error(409, str(exc), "DEPLOYMENT_SELECTION_INVALID") from exc
    return APIResponse(data=selection)


@router.put("/selection", response_model=APIResponse[DeploymentSelection])
def save_selection(
    selection_request: DeploymentValidationRequest,
    request: Request,
    service: DeploymentService = Depends(get_deployment_service),
) -> APIResponse[DeploymentSelection]:
    """Persist a deployable model/generation selection."""

    selected_by = request.headers.get("X-User") or request.headers.get("X-Client-ID")
    try:
        selection = service.save_selection(selection_request, selected_by=selected_by)
    except ModelNotFoundError as exc:
        raise _http_error(404, str(exc), "MODEL_NOT_FOUND") from exc
    except GenerationNotFoundError as exc:
        raise _http_error(404, str(exc), "GENERATION_NOT_FOUND") from exc
    except DeploymentValidationError as exc:
        raise _http_error(400, str(exc), "DEPLOYMENT_VALIDATION_ERROR") from exc
    return APIResponse(data=selection, message="Deployment selection saved")


@router.delete("/selection", response_model=APIResponse[dict[str, str]])
def clear_selection(
    service: DeploymentService = Depends(get_deployment_service),
) -> APIResponse[dict[str, str]]:
    """Clear the persisted deployment selection."""

    try:
        service.clear_selection()
    except DeploymentServiceError as exc:
        raise _http_error(500, str(exc), "DEPLOYMENT_SERVICE_ERROR") from exc
    return APIResponse(
        data={"status": "cleared"},
        message="Deployment selection cleared",
    )


__all__ = ["router"]
