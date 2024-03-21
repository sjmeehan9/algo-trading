from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete
import logging
import numpy as np

class TradingEnv(Env):
    DEFAULT_SPACE_MIN = 0
    DEFAULT_SPACE_MAX = 10000

    def __init__(self, state_builder: object):
        self.logger = logging.getLogger(__name__)

        self.state_builder = state_builder

        # Actions we can take
        num_actions = self.state_builder.pipeline['pipeline']['env_config']['action_space_size']

        self.action_space = Discrete(num_actions)

        # Observation space
        self.observation_space = self._create_obs_space()


    def _create_obs_space(self) -> None:
        space_dict = {}
        
        # Create the obs space from the StateBuilder data
        for key, value in self.state_builder.pipeline['pipeline']['model_data_config']['columns'].items():
            if value[0]==True:
               continue
            elif value[1]==True:
                space_dict[key] = Box(low=min(self.state_builder.state[key]), 
                                      high=max(self.state_builder.state[key]), 
                                      shape=(len(self.state_builder.state[key]),), 
                                      dtype=np.float32)
            else:
                space_dict[key] = Box(low=min(self.DEFAULT_SPACE_MIN), 
                                      high=max(self.DEFAULT_SPACE_MAX), 
                                      shape=(len(self.state_builder.state[key]),), 
                                      dtype=np.int32)
        
        if self.state_builder.custom_variables:
            for key, value in self.state_builder.custom_variables.items():
                space_dict[key] = Box(low=value[0], 
                                      high=value[1], 
                                      shape=(len(self.state_builder.state[key]),), 
                                      dtype=value[2])
        
        return Dict(space_dict)


    def step(self, action: int) -> tuple:
        # Placeholder for step function
        pass


    def render(self, mode: str = 'human') -> None:
        # Placeholder for render function
        pass


    def reset(self) -> dict:
        # Placeholder for reset function
        pass


    def close(self) -> None:
        # Placeholder for close function
        pass
