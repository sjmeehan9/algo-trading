import logging
from threading import Timer
from ..data_sourcing.live_streaming import LiveData
from ..data_sourcing.stream_faker import StreamFaker

class Trading:
    def __init__(self, config: dict, pipeline: dict, mode: str):
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.pipeline = pipeline
        self.mode = mode


    def start(self) -> None:
        self.logger.info('Algo trading session started')

        if self.mode == 'real':
            data_stream = LiveData(self.config, self.pipeline)
            data_stream.connect(self.config['ip_address'], self.config['port'], 0)
            Timer(data_stream.timer, data_stream.stop).start()
        elif self.mode == 'fake':
            data_stream = StreamFaker(self.config, self.pipeline)
        else:
            raise ValueError('Invalid mode')
        
        data_stream.run()

        return None