from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete
import logging

class TradingEnv(Env):
    def __init__(self, state_builder: object):
        self.logger = logging.getLogger(__name__)

        # Actions we can take
        num_actions = state_builder.pipeline['pipeline']['env_config']['action_space_size']

        self.action_space = Discrete(num_actions)
        
        # Create the obs space from the StateBuilder data
        

        self.state_builder = state_builder


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
