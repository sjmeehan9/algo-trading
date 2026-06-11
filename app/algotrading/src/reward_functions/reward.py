"""Reward factory for selecting reward calculators."""

import logging

from algotrading.src.reward_functions.profit_seeker import ProfitSeeker
from algotrading.src.reward_functions.risk_adjusted import RiskAdjusted
from algotrading.src.reward_functions.sharpe_reward import SharpeReward

logger = logging.getLogger(__name__)


def reward_factory(reward_name: str, config: dict, pipeline: dict) -> object:
    """Return the configured reward calculator instance."""

    if reward_name == "profit_seeker":
        return ProfitSeeker(config, pipeline)

    if reward_name == "risk_adjusted":
        return RiskAdjusted(config, pipeline)

    if reward_name == "sharpe_reward":
        return SharpeReward(config, pipeline)

    logger.error("reward_name not recognised")
    return None
