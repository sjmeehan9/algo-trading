import logging
import numpy as np
from ..trading.financials import Financials

class ProfitSeeker(Financials):
    CUSTOM_VARIABLES = {
        'current_position': [0, 2, np.int32],
        'trade_change': [0, 10000, np.float32],
        'running_profit': [-10000, 10000, np.float32]
    }
    PRICE_PAID = 0.0
    SET_PROFIT = 0

    def __init__(self, config: dict, pipeline: dict):
        super().__init__()

        self.logger = logging.getLogger(__name__)


    # Dictionary of custom reward variables set to initial values
    def initial_reward_variables(self) -> dict:
        reward_variables = {k: 0 for k in self.CUSTOM_VARIABLES.keys()}
        return reward_variables
    

    def reset_env_globals(self) -> None:
        self.PRICE_PAID = 0
        self.SET_PROFIT = 0
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
        self.price_paid_buy = self.PRICE_PAID / self.commision_factor
        self.price_paid_sell = self.PRICE_PAID * self.commision_factor

        price_dict = {
            'hold_nothing': self.PRICE_PAID,
            'hold_long_position': self.PRICE_PAID,
            'hold_short_position': self.PRICE_PAID,
            'buy_long': self.strike_price,
            'sell_short': self.strike_price,
            'sell_position': 0.0,
            'buyback_short': 0.0,
            'false_buy': self.PRICE_PAID,
            'false_sell': self.PRICE_PAID
        }

        self.PRICE_PAID = price_dict[self.action_type]

        return None
    
    
    def trade_change_update(self, reward_variable_dict: dict) -> dict:
        price_dict = {
            'hold_nothing': 0.0,
            'hold_long_position': (self.new_sell / self.price_paid_buy - 1) * 100,
            'hold_short_position': (self.price_paid_sell / self.new_buy - 1) * 100,
            'buy_long': (self.new_sell / self.strike_buy - 1) * 100,
            'sell_short': (self.strike_sell / self.new_buy - 1) * 100,
            'sell_position': 0.0,
            'buyback_short': 0.0,
            'false_buy': (self.new_sell / self.price_paid_buy - 1) * 100,
            'false_sell': (self.price_paid_sell / self.new_buy - 1) * 100
        }

        reward_variable_dict['trade_change'] = np.append(reward_variable_dict['trade_change'], price_dict[self.action_type])

        return reward_variable_dict
    

    def running_profit_update(self, reward_variable_dict: dict) -> dict:
        session_profit = reward_variable_dict['running_profit'][-1]
        local_profit = self.SET_PROFIT
        trade_change = reward_variable_dict['trade_change'][-1]
        last_trade_change = reward_variable_dict['trade_change'][-2]

        set_profit_dict = {
            'hold_nothing': local_profit,
            'hold_long_position': local_profit,
            'hold_short_position': local_profit,
            'buy_long': session_profit,
            'sell_short': session_profit,
            'sell_position': 0.0,
            'buyback_short': 0.0,
            'false_buy': local_profit,
            'false_sell': local_profit
        }

        self.SET_PROFIT = set_profit_dict[self.action_type]

        state_update_dict = {
            'hold_nothing': session_profit,
            'hold_long_position': local_profit + trade_change,
            'hold_short_position': local_profit + trade_change,
            'buy_long': self.SET_PROFIT + trade_change,
            'sell_short': self.SET_PROFIT + trade_change,
            'sell_position': local_profit + last_trade_change,
            'buyback_short': local_profit + last_trade_change,
            'false_buy': local_profit + trade_change,
            'false_sell': local_profit + trade_change
        }

        state_update = state_update_dict[self.action_type]

        reward_variable_dict['running_profit'] = np.append(reward_variable_dict['running_profit'], state_update)

        return reward_variable_dict