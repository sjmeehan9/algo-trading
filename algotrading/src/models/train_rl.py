import logging
from ..data_sourcing.state_builder import StateBuilder
from ..envs.trading_env import TradingEnv
from..reward_functions.profit_seeker import ProfitSeeker

class TrainRL:
    def __init__(self, config: dict, pipeline: dict):
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.pipeline = pipeline

        # Instanciate StateBuilder object
        self.state_builder = StateBuilder(self.config, self.pipeline)

        self.reward_name = self.pipeline['pipeline']['model']['model_reward']
        self.env_name = self.pipeline['pipeline']['env_config']['env_name']

        # Build a factory function to instanciate the model type
        # The model type is selected in the pipeline config


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


    def train_ppo(self):
        pass


    def start(self):
        # Setup the data feed
        self.state = self.data_setup()

        # Instanciate reward function object
        self.reward = self.reward_factory(self.reward_name)

        reward_variables, custom_variables = self.reward.initial_reward_variables()

        # Contruct initial state dictionary
        self.state_builder.initialise_state(reward_variables, custom_variables)

        # Instanciate environment object
        self.env = self.env_factory(self.env_name)
