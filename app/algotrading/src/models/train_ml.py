import logging
import os
from .train_rl import TrainRL
from ..utils import write_audit_json

class TrainML:
    AUDIT_FILENAME = 'training_sessions.json'

    def __init__(self, config: dict, pipeline: dict, evaluate: bool = False):
        self.logger = logging.getLogger(__name__)
        
        self.config = config
        self.pipeline = pipeline
        self.evaluate = evaluate

        self.path_dict = self._path_setup()


    def _path_setup(self) -> dict:
        path_dict = {}

        data_path = self.config['data_path']
        saved_data_path = os.path.join(data_path, 'models/')
        path_dict['saved_data_path'] = saved_data_path

        self.pipeline_name = self.pipeline['pipeline']['filename']
        pipeline_data_path = os.path.join(saved_data_path, f'{self.pipeline_name}/')
        path_dict['pipeline_data_path'] = pipeline_data_path

        tensorboard_path = os.path.join(pipeline_data_path, 'tensorboard/')
        path_dict['tensorboard_path'] = tensorboard_path

        backtest_data_path = os.path.join(data_path, 'backtest/')
        path_dict['backtest_data_path'] = backtest_data_path

        pipeline_backtest_path = os.path.join(backtest_data_path, f'{self.pipeline_name}/')
        path_dict['pipeline_backtest_path'] = pipeline_backtest_path

        # Create log and data folders
        if not os.path.exists(saved_data_path):
            os.makedirs(saved_data_path)

        if not os.path.exists(pipeline_data_path):
            os.makedirs(pipeline_data_path)

        if not os.path.exists(tensorboard_path):
            os.makedirs(tensorboard_path)

        if not os.path.exists(backtest_data_path):
            os.makedirs(backtest_data_path)

        if not os.path.exists(pipeline_backtest_path):
            os.makedirs(pipeline_backtest_path)

        self.input_filename = self.config['input_model']
        model_file_ext = self.pipeline['pipeline']['model']['file_extension']

        self.input_filepath = os.path.join(pipeline_data_path, self.input_filename + model_file_ext)
        path_dict['input_filepath'] = self.input_filepath

        self.model_filename = self.config['save_to_file']

        # Check if filename exists in pipeline_data_path
        self.model_filepath = os.path.join(pipeline_data_path, self.model_filename)
        path_dict['model_filepath'] = self.model_filepath

        self.audit_filepath = os.path.join(pipeline_data_path, self.AUDIT_FILENAME)
        path_dict['audit_filepath'] = self.audit_filepath

        # Run task specific setup
        if self.evaluate:
            self.eval_filename = self.config['backtest_model']
            model_file_ext = self.pipeline['pipeline']['model']['file_extension']

            self.eval_filepath = os.path.join(pipeline_data_path, self.eval_filename + model_file_ext)
            path_dict['eval_filepath'] = self.eval_filepath

            if os.path.exists(self.eval_filepath):
                self.logger.info(f'Input model found at: {self.eval_filepath}')
            else:
                raise FileNotFoundError(f'Input model not found at: {self.eval_filepath}')
        else:
            if os.path.exists(self.model_filepath + model_file_ext):
                # Ask user if they want to overwrite the file
                overwrite = input(f'{self.model_filename} already exists. Do you want to overwrite it? (y/n) ')
                if overwrite == 'n':
                    raise FileExistsError(f'{self.model_filename} already exists')
            
        return path_dict


    def session_info(self) -> None:
        session = {
            'pipeline': self.pipeline_name,
            'model': self.model_filename
        }

        session = self.training.write_session_info(session)
        write_audit_json(self.audit_filepath, session)

        return None
    

    def training_factory(self, pipeline_type: str) -> object:
        if pipeline_type == 'rl':
            return TrainRL(self.config, self.pipeline, self.path_dict, self.evaluate)
        else:
            pass


    def start(self) -> None:
        # Instanciate training object
        self.training = self.training_factory(self.pipeline['pipeline']['model']['pipeline_type'])

        # Start training
        self.training.start()

        if not self.evaluate:
            # Record training session info
            self.session_info()

        return None
        