"""Data source interfaces and implementations for the data pipeline."""

from algotrading.src.data_pipeline.sources.base import DataSource
from algotrading.src.data_pipeline.sources.broker_source import (
    BrokerDataSource,
    bar_to_record,
    contract_spec_from_config,
)
from algotrading.src.data_pipeline.sources.exceptions import (
    DataNotFoundError,
    DataSourceConnectionError,
    DataSourceError,
    DataValidationError,
)
from algotrading.src.data_pipeline.sources.file_source import FileSource
from algotrading.src.data_pipeline.sources.mock_news_source import (
    MockNewsSource,
    NewsGenerator,
)
from algotrading.src.data_pipeline.sources.news_source import NewsDataSource
from algotrading.src.data_pipeline.types import NewsQuery, NewsRecord

__all__ = [
    "DataSource",
    "FileSource",
    "BrokerDataSource",
    "bar_to_record",
    "contract_spec_from_config",
    "DataSourceError",
    "DataSourceConnectionError",
    "DataValidationError",
    "DataNotFoundError",
    "NewsDataSource",
    "MockNewsSource",
    "NewsGenerator",
    "NewsRecord",
    "NewsQuery",
]
