import logging
import os
from .train_rl import TrainRL
from ..utils import write_audit_json

class TrainML:
    AUDIT_FILENAME = 'training_sessions.json'

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

        tensorboard_path = os.path.join(pipeline_data_path, 'tensorboard/')
        path_dict['tensorboard_path'] = tensorboard_path

        # Create log and data folders
        if not os.path.exists(saved_data_path):
            os.makedirs(saved_data_path)

        if not os.path.exists(pipeline_data_path):
            os.makedirs(pipeline_data_path)

        if not os.path.exists(tensorboard_path):
            os.makedirs(tensorboard_path)

        self.input_filename = self.config['input_model']

        self.input_filepath = os.path.join(pipeline_data_path, self.input_filename)
        path_dict['input_filepath'] = self.input_filepath

        self.model_filename = self.config['save_to_file']

        # Check if filename exists in pipeline_data_path
        self.model_filepath = os.path.join(pipeline_data_path, self.model_filename)
        path_dict['model_filepath'] = self.model_filepath

        if os.path.exists(self.model_filepath):
            # Ask user if they want to overwrite the file
            overwrite = input(f'{self.model_filename} already exists. Do you want to overwrite it? (y/n) ')
            if overwrite == 'n':
                raise FileExistsError(f'{self.model_filename} already exists')
            
        self.audit_filepath = os.path.join(pipeline_data_path, self.AUDIT_FILENAME)
        path_dict['audit_filepath'] = self.audit_filepath

        return path_dict


    def session_info(self) -> dict:
        session_info = {
            'pipeline': self.pipeline_name,
            'model': self.model_filename
        }

        return session_info
    

    def training_factory(self, pipeline_type: str) -> object:
        if pipeline_type == 'rl':
            return TrainRL(self.config, self.pipeline, self.path_dict)
        else:
            pass


    def start(self) -> None:
        session_info = self.session_info()

        # Write relevant training session info to audit file
        write_audit_json(self.audit_filepath, session_info)

        # Instanciate training object
        self.training = self.training_factory(self.pipeline['pipeline']['model']['pipeline_type'])

        # Start training
        self.training.start()
        