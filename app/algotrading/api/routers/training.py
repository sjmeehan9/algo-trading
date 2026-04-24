"""REST endpoints controlling asynchronous training jobs."""

from __future__ import annotations

from algotrading.api.schemas.common import APIResponse
from algotrading.api.schemas.training import (
    TrainingJob,
    TrainingJobCreate,
    TrainingJobStatus,
)
from algotrading.api.services.training_service import (
    TrainingJobNotFoundError,
    TrainingJobStateError,
    TrainingService,
    TrainingServiceError,
    get_training_service,
)
from fastapi import APIRouter, Depends, HTTPException, Query, status

router = APIRouter(prefix="/training", tags=["training"])


def _http_error(
    status_code: int,
    error: str,
    error_code: str,
    details: dict[str, object] | None = None,
) -> HTTPException:
    """Build an HTTPException with the standard API error payload."""

    return HTTPException(
        status_code=status_code,
        detail={
            "error": error,
            "error_code": error_code,
            "details": details,
        },
    )


@router.post(
    "/jobs",
    response_model=APIResponse[TrainingJob],
    status_code=status.HTTP_201_CREATED,
)
async def create_training_job(
    request: TrainingJobCreate,
    service: TrainingService = Depends(get_training_service),
) -> APIResponse[TrainingJob]:
    """Enqueue a new training job for asynchronous execution."""

    try:
        job = await service.create_job(request)
    except TrainingServiceError as exc:
        raise _http_error(400, str(exc), "TRAINING_VALIDATION_ERROR") from exc

    return APIResponse(data=job, message="Training job queued")


@router.get("/jobs", response_model=APIResponse[list[TrainingJob]])
async def list_training_jobs(
    job_status: TrainingJobStatus | None = Query(default=None, alias="status"),
    model_id: str | None = Query(default=None),
    service: TrainingService = Depends(get_training_service),
) -> APIResponse[list[TrainingJob]]:
    """List training jobs with optional filters."""

    jobs = await service.list_jobs(status=job_status, model_id=model_id)
    return APIResponse(data=jobs)


@router.get("/jobs/{job_id}", response_model=APIResponse[TrainingJob])
async def get_training_job(
    job_id: str,
    service: TrainingService = Depends(get_training_service),
) -> APIResponse[TrainingJob]:
    """Return the current state of one training job."""

    try:
        job = await service.get_job(job_id)
    except TrainingJobNotFoundError as exc:
        raise _http_error(404, str(exc), "TRAINING_JOB_NOT_FOUND") from exc

    return APIResponse(data=job)


@router.post(
    "/jobs/{job_id}/cancel",
    response_model=APIResponse[TrainingJob],
)
async def cancel_training_job(
    job_id: str,
    service: TrainingService = Depends(get_training_service),
) -> APIResponse[TrainingJob]:
    """Cancel a queued or running training job."""

    try:
        job = await service.cancel_job(job_id)
    except TrainingJobNotFoundError as exc:
        raise _http_error(404, str(exc), "TRAINING_JOB_NOT_FOUND") from exc
    except TrainingJobStateError as exc:
        raise _http_error(409, str(exc), "TRAINING_JOB_STATE_ERROR") from exc

    return APIResponse(data=job, message="Training job cancellation requested")


__all__ = ["router"]
