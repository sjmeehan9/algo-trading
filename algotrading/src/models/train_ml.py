import logging
import os
from .train_rl import TrainRL
from ..utils import check_audit_json

class TrainML:
    AUDIT_FILENAME = 'training_sessions.json'

    def __init__(self, config: dict, pipeline: dict):
        self.logger = logging.getLogger(__name__)
        
        self.config = config
        self.pipeline = pipeline

        data_path = self.config['data_path']
        saved_data_path = os.path.join(data_path, 'models/')
        self.pipeline_name = self.pipeline['pipeline']['filename']
        pipeline_data_path = os.path.join(saved_data_path, f'{self.pipeline_name}/')

        # Create log and data folders
        if not os.path.exists(saved_data_path):
            os.makedirs(saved_data_path)

        if not os.path.exists(pipeline_data_path):
            os.makedirs(pipeline_data_path)

        self.model_filename = self.config['save_to_file']

        # Check if filename exists in pipeline_data_path
        self.model_filepath = os.path.join(pipeline_data_path, self.model_filename)
        if os.path.exists(self.model_filepath):
            # Ask user if they want to overwrite the file
            overwrite = input(f'{self.model_filename} already exists. Do you want to overwrite it? (y/n) ')
            if overwrite == 'n':
                raise FileExistsError(f'{self.model_filename} already exists')
            
        self.audit_filepath = os.path.join(pipeline_data_path, self.AUDIT_FILENAME)


    def start(self) -> None:
        #TODO
        # Write relevant training session info to audit file
        check_audit_json(self.audit_filepath)

        pipeline_type = self.pipeline['pipeline']['model']['pipeline_type']

        if pipeline_type == 'rl':
            rl_train = TrainRL(self.config, self.pipeline)
        else:
            pass
        