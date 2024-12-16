import logging
import os
from stable_baselines3 import PPO, DQN
from ..strategies.custom_logic import custom_logic_factory
from ..utils import pipeline_type_config

class Predict:
    CONFIG_FILENAME = 'pipeline_types.yml'

    def __init__(self, config: dict, pipeline: dict):
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.pipeline = pipeline

        self.pipeline_type = self.pipeline['pipeline']['pipeline_type']

        self.pipeline_type_dict = pipeline_type_config(self.CONFIG_FILENAME)

        self.get_action = self.client(self.pipeline_type)


    def client(self, pipeline_type: str) -> object:
        if pipeline_type in self.pipeline_type_dict['pipeline_type']['ml']:
            self.predictor = self.load_model()
            return self.model_predict
        elif pipeline_type in self.pipeline_type_dict['pipeline_type']['strategies']:
            self.predictor = self.load_strategy()
            return self.strategy_predict
        else:
            self.logger.error('Pipeline type not recognised')
            raise NotImplementedError('Pipeline type not recognised')


    def load_strategy(self) -> object:
        strategy_name = self.pipeline['pipeline']['strategy']['strategy_name']
        custom_logic = custom_logic_factory(strategy_name, self.config, self.pipeline)
        self.logger.info(f'Loaded strategy: {strategy_name}')

        self._states = {}

        return custom_logic


    def load_model(self) -> object:
        data_path = self.config['data_path']
        pipeline_name = self.pipeline['pipeline']['filename']
        model_filename = self.config['trading_model']
        model_file_ext = self.pipeline['pipeline']['model']['file_extension']

        model_filepath = os.path.join(data_path, 'models/', pipeline_name, model_filename + model_file_ext)

        model_type = self.pipeline['pipeline']['model']['model_type']

        if model_type == 'ppo':
            model = PPO.load(model_filepath)
            self.logger.info(f'Loaded PPO model from {model_filepath}')
        elif model_type == 'dqn':
            model = DQN.load(model_filepath)
            self.logger.info(f'Loaded DQN model from {model_filepath}')
        else:
            self.logger.error('model_type not recognised')
        
        return model


    def strategy_predict(self, obs: dict) -> tuple:
        action, self._states = self.predictor.predict(obs, self._states)
        return action, self._states


    def model_predict(self, obs: dict) -> tuple:
        action, _states = self.predictor.predict(obs)
        return action, _states
