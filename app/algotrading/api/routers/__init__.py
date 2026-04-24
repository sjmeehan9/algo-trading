"""API route modules for versioned endpoints."""

from algotrading.api.routers.generations import router as generations_router
from algotrading.api.routers.models import router as models_router
from algotrading.api.routers.strategies import router as strategies_router

__all__ = ["models_router", "strategies_router", "generations_router"]
