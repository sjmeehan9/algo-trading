import logging
import os
from .train_ml import TrainML

class BackTest:
    def __init__(self, config: dict, pipeline: dict):
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.pipeline = pipeline

        self.pipeline_type = self.pipeline['pipeline']['pipeline_type']

        self.path_dict = self._path_setup()

    
    def _path_setup(self) -> dict:
        path_dict = {}

        data_path = self.config['data_path']
        backtest_data_path = os.path.join(data_path, 'backtest/')
        path_dict['backtest_data_path'] = backtest_data_path

        pipeline_backtest_path = os.path.join(backtest_data_path, f'{self.pipeline_name}/')
        path_dict['pipeline_backtest_path'] = pipeline_backtest_path

        if not os.path.exists(backtest_data_path):
            os.makedirs(backtest_data_path)

        if not os.path.exists(pipeline_backtest_path):
            os.makedirs(pipeline_backtest_path)

        return path_dict


    def client(self, pipeline_type: str) -> object:
        if pipeline_type in TrainML.ML_TYPES:
            return TrainML(self.config, self.pipeline, self.path_dict, True)
        elif pipeline_type == 'strategy':
            raise NotImplementedError('Strategy pipeline is not yet supported')
        else:
            self.logger.error('Pipeline type not recognised')
            raise NotImplementedError('Pipeline type not recognised')


    def start(self) -> None:
        self.logger.info('Backtest initiated')

        client = self.client(self.pipeline_type)

        client.start()

        self.logger.info('Backtest completed')

        return None