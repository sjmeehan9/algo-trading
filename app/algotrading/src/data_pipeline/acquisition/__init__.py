"""Historical data acquisition services for training and backtesting."""

from algotrading.src.data_pipeline.acquisition.market_data import (
    AcquisitionResult,
    AcquisitionStatus,
    HistoricalMarketDataAcquirer,
    MarketDataAcquisitionCancelled,
    MarketDataAcquisitionError,
    SymbolAcquisitionReport,
)
from algotrading.src.data_pipeline.acquisition.news_data import (
    HistoricalNewsAcquirer,
    NewsDataAcquisitionError,
    model_requires_news,
)

__all__ = [
    "AcquisitionResult",
    "AcquisitionStatus",
    "HistoricalMarketDataAcquirer",
    "HistoricalNewsAcquirer",
    "MarketDataAcquisitionCancelled",
    "MarketDataAcquisitionError",
    "NewsDataAcquisitionError",
    "SymbolAcquisitionReport",
    "model_requires_news",
]
