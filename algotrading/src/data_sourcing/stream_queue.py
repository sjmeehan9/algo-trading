import logging

class StreamQueue:
    def __init__(self, config: dict, pipeline: dict):
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.pipeline = pipeline


    def put(self, bar: dict) -> None:
        self.logger.info(f'Putting bar {bar} in queue')

        return None
    

    def route(self) -> None:

        return None