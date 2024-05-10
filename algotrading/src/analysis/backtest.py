import logging
import os

class BackTest:
    def __init__(self, config: dict, pipeline: dict):
        self.logger = logging.getLogger(__name__)
        
        self.config = config
        self.pipeline = pipeline

        self.path_dict = self._path_setup()


    def _path_setup(self) -> dict:
        path_dict = {}

        data_path = self.config['data_path']
        saved_data_path = os.path.join(data_path, 'models/')
        path_dict['saved_data_path'] = saved_data_path

        self.pipeline_name = self.pipeline['pipeline']['filename']
        pipeline_data_path = os.path.join(saved_data_path, f'{self.pipeline_name}/')
        path_dict['pipeline_data_path'] = pipeline_data_path

        # Check if path exists, if not raise an error
        if not os.path.exists(saved_data_path):
            raise FileNotFoundError(f'{saved_data_path} does not exist, no input model will be found')

        if not os.path.exists(pipeline_data_path):
            raise FileNotFoundError(f'{pipeline_data_path} does not exist, no input model will be found')

        self.input_filename = self.config['test_model']
        model_file_ext = self.pipeline['pipeline']['model']['file_extension']

        self.input_filepath = os.path.join(pipeline_data_path, self.input_filename + model_file_ext)
        path_dict['input_filepath'] = self.input_filepath

        if os.path.exists(self.input_filepath):
            self.logger.info(f'Input model found at: {self.input_filepath}')
        else:
            raise FileNotFoundError(f'Input model not found at: {self.input_filepath}')
        
        backtest_data_path = os.path.join(data_path, 'backtest/')
        path_dict['backtest_data_path'] = backtest_data_path

        pipeline_backtest_path = os.path.join(backtest_data_path, f'{self.pipeline_name}/')
        path_dict['pipeline_backtest_path'] = pipeline_backtest_path

        if not os.path.exists(backtest_data_path):
            os.makedirs(backtest_data_path)

        if not os.path.exists(pipeline_backtest_path):
            os.makedirs(pipeline_backtest_path)

        return path_dict