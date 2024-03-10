import logging

class TradingEnv:
    def __init__(self, config: dict, pipeline: dict):
        self.logger = logging.getLogger(__name__)

        # Add in Gym stuff
        
        # Create the obs space from the StateBuilder data

