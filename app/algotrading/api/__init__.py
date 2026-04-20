"""FastAPI backend entrypoints for the algo-trading web API."""

from algotrading.api.config import APIConfig
from algotrading.api.main import create_app

__all__ = ["APIConfig", "create_app"]
