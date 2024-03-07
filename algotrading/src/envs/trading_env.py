import logging
from typing import Dict, List, Union

class TradingEnv:
    def __init__(self, config: Dict[str, Union[str, List[str], Dict[str, Union[str, int]]]], pipeline: Dict[str, Union[str, Dict[str, str]]]):
        self.logger = logging.getLogger(__name__)

        # Add in Gym stuff
        
        # Create the obs space from the StateBuilder data

