import logging
from .profit_seeker import ProfitSeeker

logger = logging.getLogger(__name__)

def reward_factory(reward_name: str, config: dict, pipeline: dict) -> object:
    if reward_name == 'profit_seeker':
        return ProfitSeeker(config, pipeline)
    else:
        logger.error('reward_name not recognised')
        return None