import logging
from .profit_metrics import ProfitMetrics

logger = logging.getLogger(__name__)

def custom_logic_factory(strategy_name: str, config: dict, pipeline: dict) -> object:
    if strategy_name == 'profit_metrics':
        return ProfitMetrics(config, pipeline)
    else:
        logger.error('strategy_name not recognised')
        return None