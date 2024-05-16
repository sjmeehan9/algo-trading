from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete
import logging
import numpy as np

class TradingEnv(Env):
    DEFAULT_SPACE_MIN = 0
    DEFAULT_SPACE_MAX = 10000
    ACTION_SPACE_SIZE = 3

    def __init__(self, state_builder: object):
        super().__init__()

        self.logger = logging.getLogger(__name__)

        self.state_builder = state_builder

        # Actions we can take
        self.action_space = Discrete(self.ACTION_SPACE_SIZE)

        # Observation space
        self.observation_space = self._create_obs_space()


    def _create_obs_space(self) -> None:
        space_dict = {}

        custom_variables = self.state_builder.reward.CUSTOM_VARIABLES
        
        # Create the obs space from the StateBuilder data
        for key, value in self.state_builder.pipeline['pipeline']['model_data_config']['columns'].items():
            if value[0]==True:
               continue
            elif value[1]==True:
                space_dict[key] = Box(low=min(self.state_builder.state[key]), 
                                      high=max(self.state_builder.state[key]), 
                                      shape=self.state_builder.state[key].shape, 
                                      dtype=np.float64)
            else:
                space_dict[key] = Box(low=self.DEFAULT_SPACE_MIN, 
                                      high=self.DEFAULT_SPACE_MAX, 
                                      shape=self.state_builder.state[key].shape, 
                                      dtype=np.float64)
        
        if custom_variables:
            for key, value in custom_variables.items():
                space_dict[key] = Box(low=value[0], 
                                      high=value[1], 
                                      shape=self.state_builder.state[key].shape, 
                                      dtype=value[2])
        
        return Dict(space_dict)


    def step(self, action: int) -> tuple:
        self.logger.info('step taken')
        
        self.state_builder.state_step(action)

        # Within the reward function, calculate the reward for the current step
        reward = self.state_builder.reward.calculate_reward(self.state_builder.state)
        
        info = {}

        self.logger.info(f'new step number: {self.state_builder.state_counters["step"]}')

        return self.state_builder.state, reward, self.state_builder.terminated, False, info


    def render(self, mode: str = 'human') -> None:
        pass


    def reset(self, seed=None, options=None) -> dict:
        self.logger.info('env reset')

        self.state_builder.state_counters['step'] = 0

        self.logger.info(f'terminated {self.state_builder.terminated}')
        self.logger.info(f'timed_out {self.state_builder.timed_out}')
        self.logger.info(f'new step number {self.state_builder.state_counters["step"]}')

        if self.state_builder.terminated:
            self.state_builder.update_episode_counter()
            self.state_builder.reward.reset_env_globals()

        if not self.state_builder.timed_out:
            self.state_builder.initialise_state(self.state_builder.reward)

        info = {}

        return self.state_builder.state, info


    def close(self) -> None:
        return None
