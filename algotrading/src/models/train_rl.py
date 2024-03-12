import logging
from ..data_sourcing.state_builder import StateBuilder

class TrainRL():
    def __init__(self, config: dict, pipeline: dict):
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.pipeline = pipeline

        # Build a function that will create the environment
        # Use def env_factory(env_name): which will if else through the different envs
        # Uses the StateBuilder object and the ep length
        # The type of env is selected in the pipeline config

        # Build a factory function to instanciate the reward function
        # The reward function is selected in the pipeline config

        # Build a factory function to instanciate the model type
        # The model type is selected in the pipeline config


    def train_ppo(self):
        pass


    def data_setup(self) -> None:
        if self.config['data_mode'] == 'historical':
            self.state_builder.read_data()
            self.logger.info('placeholder')
        elif self.config['data_mode'] == 'live':
            # Placeholder for live data stream setup
            # For live data, pass the data through to StateBuilder object
            pass
        else:
            self.logger.error('data_mode not recognised')

        return None
    

    def env_factory(self, env_name: str):
        pass


    def start(self):
        # Instanciate StateBuilder object
        self.state_builder = StateBuilder(self.config, self.pipeline)

        # Setup the data feed
        self.data_setup()
