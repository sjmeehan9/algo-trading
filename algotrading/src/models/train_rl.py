import logging
from ..data_sourcing.state_builder import StateBuilder
from ..envs.trading_env import TradingEnv

class TrainRL():
    def __init__(self, config: dict, pipeline: dict):
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.pipeline = pipeline

        # Instanciate StateBuilder object
        self.state_builder = StateBuilder(self.config, self.pipeline)        

        # Build a factory function to instanciate the reward function
        # The reward function is selected in the pipeline config

        # Build a factory function to instanciate the model type
        # The model type is selected in the pipeline config


    def train_ppo(self):
        pass


    def data_setup(self) -> dict:
        if self.config['data_mode'] == 'historical':
            self.state_builder.read_data()
            self.logger.info('Data read from historical files')

            # Contruct initial state dictionary
            state = self.state_builder.initialise_state()
            return state 
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


    def start(self):
        # Setup the data feed
        self.state = self.data_setup()

        # Instanciate reward function object

        # Instanciate environment object
        env_name = self.pipeline['pipeline']['env_config']['env_name']

        self.env = self.env_factory(env_name)
