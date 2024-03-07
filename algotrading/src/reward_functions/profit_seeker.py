import logging
from typing import Dict, List, Union

class ProfitSeeker:
    def __init__(self, config: Dict[str, Union[str, List[str], Dict[str, Union[str, int]]]], pipeline: Dict[str, Union[str, Dict[str, str]]]):
        self.logger = logging.getLogger(__name__)

        # Class variables for custom reward variables

        # Build function with calc instructions for each custom reward variable
