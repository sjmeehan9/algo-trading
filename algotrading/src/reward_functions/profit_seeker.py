import logging
import numpy as np
from ..trading.financials import Financials

class ProfitSeeker(Financials):
    CUSTOM_VARIABLES = {
        'current_position': [0, 2, np.int32],
        'trade_change': [0, 10000, np.float32],
        'running_profit': [-10000, 10000, np.float32]
    }
    CURRENT_PRICE = 0.0
    RUN_PROFIT = 0

    def __init__(self, config: dict, pipeline: dict):
        super().__init__()

        self.logger = logging.getLogger(__name__)


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
                self.action_type = 'hold_nothing'
            elif self.current_position == 1:
                self.action_type = 'sell_position'
            elif self.current_position == 2:
                self.action_type = 'buyback_short'
        elif self.current_position == 0:
            if action == 0:
                self.action_type = 'hold_nothing'
            elif action == 1:
                self.action_type = 'buy_long'
            elif action == 2:
                self.action_type = 'sell_short'
        elif self.current_position == 1:
            if action == 0:
                self.action_type = 'hold_long_position'
            elif action == 1:
                self.action_type = 'false_buy'
            elif action == 2:
                self.action_type = 'sell_position'
        elif self.current_position == 2:
            if action == 0:
                self.action_type = 'hold_short_position'
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

        reward_variable_dict = self.current_position_update(action, reward_variable_dict)

        self.current_price_update(state_df)

        reward_variable_dict = self.trade_change_update(reward_variable_dict)

        reward_variable_dict = self.running_profit_update(reward_variable_dict)

        return reward_variable_dict
    

    def current_position_update(self, action: int, reward_variable_dict: dict) -> dict:

        position_dict = {
            'hold_nothing': self.current_position,
            'hold_long_position': self.current_position,
            'hold_short_position': self.current_position,
            'buy_long': action,
            'sell_short': action,
            'sell_position': 0,
            'buyback_short': 0,
            'false_buy': self.current_position,
            'false_sell': self.current_position
        }

        reward_variable_dict['current_position'] = np.append(reward_variable_dict['current_position'], position_dict[self.action_type])

        return reward_variable_dict
    

    def current_price_update(self, state_df: dict) -> None:
        self.strike_price = state_df['close'].iloc[-2]
        self.new_price = state_df['close'].iloc[-1]

        self.strike_buy = self.strike_price / self.commision_factor
        self.strike_sell = self.strike_price * self.commision_factor
        self.new_buy = self.new_price / self.commision_factor
        self.new_sell = self.new_price * self.commision_factor
        self.current_buy = self.CURRENT_PRICE / self.commision_factor
        self.current_sell = self.CURRENT_PRICE * self.commision_factor

        price_dict = {
            'hold_nothing': self.CURRENT_PRICE,
            'hold_long_position': self.CURRENT_PRICE,
            'hold_short_position': self.CURRENT_PRICE,
            'buy_long': self.strike_price,
            'sell_short': self.strike_price,
            'sell_position': 0.0,
            'buyback_short': 0.0,
            'false_buy': self.CURRENT_PRICE,
            'false_sell': self.CURRENT_PRICE
        }

        self.CURRENT_PRICE = price_dict[self.action_type]

        return None
    
    
    def trade_change_update(self, reward_variable_dict: dict) -> dict:
        price_dict = {
            'hold_nothing': 0.0,
            'hold_long_position': (self.new_sell / self.current_buy - 1) * 100,
            'hold_short_position': (self.current_sell / self.new_buy - 1) * 100,
            'buy_long': (self.new_sell / self.strike_buy - 1) * 100,
            'sell_short': (self.strike_sell / self.new_buy - 1) * 100,
            'sell_position': 0.0,
            'buyback_short': 0.0,
            'false_buy': (self.new_sell / self.current_buy - 1) * 100,
            'false_sell': (self.current_sell / self.new_buy - 1) * 100
        }

        reward_variable_dict['trade_change'] = np.append(reward_variable_dict['trade_change'], price_dict[self.action_type])

        return reward_variable_dict
    

    def running_profit_update(self, reward_variable_dict: dict) -> dict:
        # TODO
        # self.RUN_PROFIT is for current trade and the dict variable is overall session
        # current trade profit is just trade price added together
        # overall session profit is the sum of all trade profits

        price_dict = {
            'hold_nothing': self.RUN_PROFIT,
            'hold_long_position': self.RUN_PROFIT + (self.new_sell / self.current_buy - 1) * 100,
            'hold_short_position': self.RUN_PROFIT + (self.current_sell / self.new_buy - 1) * 100,
            'buy_long': self.RUN_PROFIT - self.strike_price,
            'sell_short': self.RUN_PROFIT + self.strike_price,
            'sell_position': 0.0,
            'buyback_short': 0.0,
            'false_buy': self.RUN_PROFIT + (self.new_price - self.CURRENT_PRICE),
            'false_sell': self.RUN_PROFIT + (self.CURRENT_PRICE - self.new_price)
        }

        self.RUN_PROFIT = price_dict[self.action_type]

        reward_variable_dict['running_profit'] = np.append(reward_variable_dict['running_profit'], self.RUN_PROFIT)

        return reward_variable_dict