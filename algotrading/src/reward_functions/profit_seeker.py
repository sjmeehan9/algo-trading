import logging
import numpy as np

class ProfitSeeker:
    CUSTOM_VARIABLES = {
        'current_position': [0, 2, np.int32],
        'trade_price': [0, 10000, np.float32],
        'running_profit': [-10000, 10000, np.float32]
    }

    def __init__(self, config: dict, pipeline: dict):
        self.logger = logging.getLogger(__name__)

        # Build function with calc instructions for each custom reward variable

    # Dictionary of custom reward variables set to initial values
    def initial_reward_variables(self) -> dict:
        reward_variables = {k: 0 for k in self.CUSTOM_VARIABLES.keys()}
        return reward_variables
