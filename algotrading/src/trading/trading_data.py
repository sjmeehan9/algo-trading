import logging
from threading import Timer
from ..data_sourcing.live_streaming import LiveData
from ..data_sourcing.stream_faker import StreamFaker

class TradingStream:
    def __init__(self, config: dict, pipeline: dict, mode: str):
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.pipeline = pipeline
        self.mode = mode


    def streamer(self) -> None:
        if self.mode == 'real':
            data_stream = LiveData(self.config, self.pipeline)
            data_stream.connect(self.config['ip_address'], self.config['port'], 0)
            Timer(data_stream.timer, data_stream.stop).start()
            return data_stream
        elif self.mode == 'fake':
            data_stream = StreamFaker(self.config, self.pipeline)
            return data_stream
        else:
            raise ValueError('Invalid mode')
        

    def start(self) -> None:
        self.logger.info('Algo trading session started')

        data_stream = self.streamer()
        
        data_stream.run()

        return None