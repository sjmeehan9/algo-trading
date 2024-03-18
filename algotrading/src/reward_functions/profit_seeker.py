import logging

class ProfitSeeker:
    CUSTOM_VARIABLES = {
        'current_position': 'what type of trade is currently open',
        'trade_price': 'price of the trade to initiate the current position',
        'running_profit': 'total running profit of the current position'
    }

    def __init__(self, config: dict, pipeline: dict):
        self.logger = logging.getLogger(__name__)

        # Build function with calc instructions for each custom reward variable

    # Dictionary of custom reward variables set to 0
    def initial_reward_variables(self) -> dict:
        return {k: 0 for k in self.CUSTOM_VARIABLES.keys()}
