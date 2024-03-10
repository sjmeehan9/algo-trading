import logging

class ProfitSeeker:
    def __init__(self, config: dict, pipeline: dict):
        self.logger = logging.getLogger(__name__)

        # Class variables for custom reward variables

        # Build function with calc instructions for each custom reward variable
