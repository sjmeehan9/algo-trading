"""Reward factory for selecting reward calculators."""

import logging

from algotrading.src.reward_functions.profit_seeker import ProfitSeeker

logger = logging.getLogger(__name__)


def reward_factory(reward_name: str, config: dict, pipeline: dict) -> object:
    """Return the configured reward calculator instance."""

    if reward_name == "profit_seeker":
        return ProfitSeeker(config, pipeline)

    logger.error("reward_name not recognised")
    return None
