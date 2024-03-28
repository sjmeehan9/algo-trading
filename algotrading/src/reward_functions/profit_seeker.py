import logging
import numpy as np

class ProfitSeeker:
    CUSTOM_VARIABLES = {
        'current_position': [0, 2, np.int32],
        'trade_price': [0, 10000, np.float32],
        'running_profit': [-10000, 10000, np.float32]
    }
    CURRENT_PRICE = 0
    RUN_PROFIT = 0

    def __init__(self, config: dict, pipeline: dict):
        self.logger = logging.getLogger(__name__)

        # Build function with calc instructions for each custom reward variable


    # Dictionary of custom reward variables set to initial values
    def initial_reward_variables(self) -> dict:
        reward_variables = {k: 0 for k in self.CUSTOM_VARIABLES.keys()}
        return reward_variables
    

    def reset_env_globals(self) -> None:
        self.CURRENT_PRICE = 0
        self.RUN_PROFIT = 0
        return None
    

    def determine_action_type(self, action: int, terminated: bool) -> None:
        # Action type dictionary
        if terminated:
            if self.current_position == 0:
                self.action_type = 'hold_position'
            elif self.current_position == 1:
                self.action_type = 'sell_position'
            elif self.current_position == 2:
                self.action_type = 'buyback_short'
        elif self.current_position == 0:
            if action == 0:
                self.action_type = 'hold_position'
            elif action == 1:
                self.action_type = 'buy_long'
            elif action == 2:
                self.action_type = 'sell_short'
        elif self.current_position == 1:
            if action == 0:
                self.action_type = 'hold_position'
            elif action == 1:
                self.action_type = 'false_buy'
            elif action == 2:
                self.action_type = 'sell_position'
        elif self.current_position == 2:
            if action == 0:
                self.action_type = 'hold_position'
            elif action == 1:
                self.action_type = 'buyback_short'
            elif action == 2:
                self.action_type = 'false_sell'
        else:
            self.logger.error('Action type not recognised')
        
        return None
    

    def reward_variable_step(self, action: int, state_df: dict, reward_variable_dict: dict, terminated: bool) -> dict:
        self.current_position = reward_variable_dict['current_position'][-1]

        self.determine_action_type(action, terminated)

        reward_variable_dict = self.next_position(action, reward_variable_dict)

        return reward_variable_dict
    

    def next_position(self, action: int, reward_variable_dict: dict) -> dict:

        position_dict = {
            'hold_position': self.current_position,
            'buy_long': action,
            'sell_short': action,
            'sell_position': 0,
            'buyback_short': 0,
            'false_buy': self.current_position,
            'false_sell': self.current_position
        }

        reward_variable_dict['current_position'] = np.append(reward_variable_dict['current_position'], position_dict[self.action_type])

        return reward_variable_dict