"""API route modules for versioned endpoints."""

from algotrading.api.routers.backtesting import router as backtesting_router
from algotrading.api.routers.deployment import router as deployment_router
from algotrading.api.routers.generations import router as generations_router
from algotrading.api.routers.models import router as models_router
from algotrading.api.routers.optimizer import router as optimizer_router
from algotrading.api.routers.strategies import router as strategies_router
from algotrading.api.routers.trading import router as trading_router
from algotrading.api.routers.training import router as training_router

__all__ = [
    "backtesting_router",
    "deployment_router",
    "models_router",
    "strategies_router",
    "generations_router",
    "training_router",
    "trading_router",
    "optimizer_router",
]
