import logging
import os
from stable_baselines3 import PPO
from ..data_sourcing.state_builder import StateBuilder
from ..envs.trading_env import TradingEnv
from ..reward_functions.profit_seeker import ProfitSeeker

class TrainRL:
    def __init__(self, config: dict, pipeline: dict, path_dict: dict):
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.pipeline = pipeline
        self.path_dict = path_dict

        # Instanciate StateBuilder object
        self.state_builder = StateBuilder(self.config, self.pipeline)

        self.reward_name = self.pipeline['pipeline']['model']['model_reward']
        self.env_name = self.pipeline['pipeline']['env_config']['env_name']
        self.model_type = self.pipeline['pipeline']['model']['model_type']
        self.model_filename = self.config['save_to_file']


    def data_setup(self) -> dict:
        if self.config['data_mode'] == 'historical':
            self.state_builder.read_data()
            self.logger.info('Data read from historical files')
            return None
        elif self.config['data_mode'] == 'live':
            # Placeholder for live data stream setup
            # For live data, pass the data through to StateBuilder object
            return None
        else:
            self.logger.error('data_mode not recognised')
            return None
    

    def env_factory(self, env_name: str) -> object:
        if env_name == 'trading_env':
            return TradingEnv(self.state_builder)
        else:
            self.logger.error('env_name not recognised')
            return None
        
    
    def reward_factory(self, reward_name: str) -> object:
        if reward_name == 'profit_seeker':
            return ProfitSeeker(self.config, self.pipeline)
        else:
            self.logger.error('reward_name not recognised')
            return None
        

    def model_factory(self, model_type: str) -> object:
        if model_type == 'ppo':
            return self.train_ppo()
        else:
            self.logger.error('model_type not recognised')
            return None


    def train_ppo(self) -> None:
        # Load input model or create new model
        if os.path.exists(self.path_dict['input_filepath']):
            self.model = PPO.load(self.path_dict['input_filepath'])
            self.logger.info(f'Loaded model from {self.path_dict["input_filepath"]}')
        else:
            self.model = PPO('MultiInputPolicy', self.env, verbose=1, tensorboard_log=self.path_dict['tensorboard_path'])
            self.logger.info('Created new model')
        
        # Train the model
        self.model.learn(total_timesteps=self.state_builder.total_timesteps)

        # Save the model
        self.model.save(self.path_dict['model_filepath'])

        # Revisit saving the session info
        #TODO

        return None


    def start(self):
        # Setup the data feed
        self.state = self.data_setup()

        # Instanciate reward function object
        self.reward = self.reward_factory(self.reward_name)

        # Contruct initial state dictionary
        self.state_builder.initialise_state(self.reward)

        # Instanciate environment object
        self.env = self.env_factory(self.env_name)

        # Run training
        self.model_factory(self.model_type)
