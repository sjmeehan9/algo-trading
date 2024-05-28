import logging
from threading import Timer
from ..data_sourcing.live_streaming import LiveData
from ..data_sourcing.stream_faker import StreamFaker
from ..data_sourcing.stream_queue import StreamQueue

class Trading:
    def __init__(self, config: dict, pipeline: dict):
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.pipeline = pipeline

        self.queue = StreamQueue(self.config, self.pipeline)


    def start(self) -> None:
        self.logger.info('Algo trading session started')

        # data_stream = StreamFaker(self.config, self.pipeline, self.queue)
        
        data_stream = LiveData(self.config, self.pipeline)
        data_stream.connect(self.config['ip_address'], self.config['port'], 0)
        Timer(data_stream.timer, data_stream.stop).start()
        
        data_stream.run()

        return None