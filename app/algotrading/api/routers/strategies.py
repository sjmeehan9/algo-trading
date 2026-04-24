"""Strategy catalog endpoints."""

from __future__ import annotations

from algotrading.api.schemas.common import APIResponse
from algotrading.api.schemas.strategies import StrategyDetail, StrategyInfo
from algotrading.api.services import (
    ModelService,
    StrategyNotFoundError,
    get_model_service,
)
from algotrading.src.models.signals import SignalType
from fastapi import APIRouter, Depends, HTTPException

router = APIRouter(prefix="/strategies", tags=["strategies"])


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


@router.get("", response_model=APIResponse[list[StrategyInfo]])
def list_strategies(
    signal_type: SignalType | None = None,
    service: ModelService = Depends(get_model_service),
) -> APIResponse[list[StrategyInfo]]:
    """List registered custom strategies."""

    return APIResponse(data=service.list_strategies(signal_type=signal_type))


@router.get("/{strategy_id}", response_model=APIResponse[StrategyDetail])
def get_strategy(
    strategy_id: str,
    service: ModelService = Depends(get_model_service),
) -> APIResponse[StrategyDetail]:
    """Get details for one strategy."""

    try:
        strategy = service.get_strategy(strategy_id)
    except StrategyNotFoundError as exc:
        raise _http_error(404, str(exc), "STRATEGY_NOT_FOUND") from exc

    return APIResponse(data=strategy)


__all__ = ["router"]
