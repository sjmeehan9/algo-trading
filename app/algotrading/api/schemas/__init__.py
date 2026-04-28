"""Schema definitions used by API request and response handlers."""

from algotrading.api.schemas.backtesting import (
    BacktestComparison,
    BacktestComparisonRequest,
    BacktestRequest,
    BacktestResult,
    BacktestStatus,
    EquityPoint,
    PerformanceMetrics,
    TradeAction,
    TradeRecord,
)
from algotrading.api.schemas.common import APIError, APIResponse, PaginatedResponse
from algotrading.api.schemas.generations import (
    GenerationComparison,
    GenerationComparisonRequest,
    GenerationDetail,
    GenerationSummary,
)
from algotrading.api.schemas.models import (
    CoreRLModelConfig,
    ModelConfigBase,
    ModelConfigCreate,
    ModelConfigResponse,
    ModelConfigUpdate,
    ModelType,
    SupportingModelConfig,
)
from algotrading.api.schemas.strategies import StrategyDetail, StrategyInfo
from algotrading.api.schemas.training import (
    TrainingJob,
    TrainingJobCreate,
    TrainingJobStatus,
    TrainingProgress,
)

__all__ = [
    "APIError",
    "APIResponse",
    "PaginatedResponse",
    "BacktestStatus",
    "TradeAction",
    "BacktestRequest",
    "TradeRecord",
    "PerformanceMetrics",
    "EquityPoint",
    "BacktestResult",
    "BacktestComparison",
    "BacktestComparisonRequest",
    "ModelType",
    "ModelConfigBase",
    "CoreRLModelConfig",
    "SupportingModelConfig",
    "ModelConfigCreate",
    "ModelConfigUpdate",
    "ModelConfigResponse",
    "StrategyInfo",
    "StrategyDetail",
    "GenerationSummary",
    "GenerationDetail",
    "GenerationComparison",
    "GenerationComparisonRequest",
    "TrainingJob",
    "TrainingJobCreate",
    "TrainingJobStatus",
    "TrainingProgress",
]
