import logging
from typing import Dict, List, Union

class TrainRL():
    def __init__(self, config: Dict[str, Union[str, List[str], Dict[str, Union[str, int]]]], pipeline: Dict[str, Union[str, Dict[str, str]]]):
        self.logger = logging.getLogger(__name__)

        # Instanciate StateBuilder object
        # Pass config and pipeline to StateBuilder object

        # Read data_mode from local config
        # If data_mode is 'historical', call the state_builder read data function
        # If data_mode is 'live', instanciate a LiveData object and setup the stream
        # For live data, pass the data through to StateBuilder object
        # Placeholder for sequence of steps to train a model with live data

        # Self variable for ep length from StateBuilder function

        # Build a function that will create the environment
        # Use def env_factory(env_name): which will if else through the different envs
        # Uses the StateBuilder object and the ep length
        # The type of env is selected in the pipeline config

        # Build a factory function to instanciate the reward function
        # The reward function is selected in the pipeline config


    def train_ppo(self):
        pass

    def start(self):
        pass