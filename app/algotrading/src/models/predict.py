import logging
import os
from stable_baselines3 import PPO, DQN

class Predict:
    def __init__(self, config: dict, pipeline: dict):
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.pipeline = pipeline

        data_path = self.config['data_path']
        pipeline_name = self.pipeline['pipeline']['filename']
        model_filename = self.config['trading_model']
        model_file_ext = self.pipeline['pipeline']['model']['file_extension']

        self.model_filepath = os.path.join(data_path, 'models/', pipeline_name, model_filename + model_file_ext)

        self.model_type = self.pipeline['pipeline']['model']['model_type']

        self.model = self.load_model()


    def load_model(self) -> object:
        if self.model_type == 'ppo':
            model = PPO.load(self.model_filepath)
            self.logger.info(f'Loaded PPO model from {self.model_filepath}')
        elif self.model_type == 'dqn':
            model = DQN.load(self.model_filepath)
            self.logger.info(f'Loaded DQN model from {self.model_filepath}')
        else:
            self.logger.error('model_type not recognised')
        
        return model
    

    def get_action(self, obs: dict) -> tuple:
        action, _states = self.model.predict(obs)
        return action, _states