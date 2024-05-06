import logging
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
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
        self.model_config = self.pipeline['pipeline']['model']['model_config']
        self.model_input = self.config['input_model']
        self.model_filename = self.config['save_to_file']


    def write_session_info(self, session: dict) -> dict:
        training_dates = [d.isoformat() for d in self.state_builder.training_date_list]

        session_info = {
            'model': self.model_filename,
            'input_model': self.model_input,
            'reward_function': self.reward_name,
            'env': self.env_name,
            'training_dates': training_dates,
            'total_timesteps': self.state_builder.total_timesteps,
            'data_mode': self.config['data_mode'],
            'episode_length': self.state_builder.episode_length
        }

        session.update(session_info)

        return session


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
            env = TradingEnv(self.state_builder)
            try:
                check_env(env)
                return env
            except Exception as e:
                self.logger.error(f'Environment check failed: {e}')
                raise e
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
            self.model = PPO('MultiInputPolicy', self.env, n_steps=self.model_config['n_steps'], batch_size=self.model_config['batch_size'], verbose=1, tensorboard_log=self.path_dict['tensorboard_path'])
            self.logger.info('Created new model')
        
        # Train the model
        self.model.learn(total_timesteps=self.state_builder.total_timesteps, tb_log_name=self.model_filename)

        # Save the model
        self.model.save(self.path_dict['model_filepath'])

        return None


    def start(self) -> None:
        # Setup the data feed
        self.data_setup()

        # Instanciate reward function object
        self.reward = self.reward_factory(self.reward_name)

        # Contruct initial state dictionary
        self.state_builder.initialise_state(self.reward)

        # Instanciate environment object
        self.env = self.env_factory(self.env_name)

        # Run training
        self.model_factory(self.model_type)

        return None
