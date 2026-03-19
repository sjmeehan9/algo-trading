"""Exception hierarchy for data pipeline sources."""

from __future__ import annotations


class DataSourceError(Exception):
    """Base exception for all data source failures."""


class DataSourceConnectionError(DataSourceError):
    """Raised when a data source cannot connect or initialize."""


class DataValidationError(DataSourceError):
    """Raised when source data fails validation checks."""


class DataNotFoundError(DataSourceError):
    """Raised when requested data cannot be located."""
