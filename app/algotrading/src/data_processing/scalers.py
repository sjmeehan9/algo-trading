import logging
from sklearn.preprocessing import MinMaxScaler

class Scaler:
    def __init__(self, pipeline: dict):
        self.logger = logging.getLogger(__name__)

        self.pipeline = pipeline

        self.scaler = self.pipeline['pipeline']['model_data_config']['scaler']


    def scaler_factory(self) -> object:
        if self.scaler == 'MinMaxScaler':
            scaler = MinMaxScaler()
            self.logger.info('Using MinMaxScaler')
            return scaler
        else:
            raise NotImplementedError('Scaler not recognised')
