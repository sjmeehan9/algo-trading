"""Strategy factory for selecting custom logic implementations."""

import logging

from algotrading.src.strategies.profit_metrics import ProfitMetrics

logger = logging.getLogger(__name__)


def custom_logic_factory(strategy_name: str, config: dict, pipeline: dict) -> object:
    """Return the configured strategy implementation instance."""

    if strategy_name == "profit_metrics":
        return ProfitMetrics(config, pipeline)

    logger.error("strategy_name not recognised")
    return None
