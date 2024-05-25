import logging
from ..data_sourcing.stream_faker import StreamFaker

class Trading:
    def __init__(self, config: dict, pipeline: dict):
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.pipeline = pipeline


    def start(self) -> None:
        self.logger.info('Algo trading session started')

        data_stream = StreamFaker(self.config, self.pipeline)
        data_stream.start()

        return None