"""Standalone data pipeline package.

This package provides foundational types and module boundaries for pipeline
data sources, processors, state management, and routing.
"""

from algotrading.src.data_pipeline.types import (
    DataBatch,
    DataFrequency,
    DataRecord,
    DataType,
    SourceMetadata,
)

__version__ = "1.0.0"

__all__ = [
    "DataBatch",
    "DataFrequency",
    "DataRecord",
    "DataType",
    "SourceMetadata",
    "__version__",
]
