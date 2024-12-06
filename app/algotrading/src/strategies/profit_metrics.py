import logging
import numpy as np
import pandas as pd
import warnings
from .strategy_wrapper import strategy_wrapper_function
from ..trading.financials import Financials
from ..trading.payload import Payload

class ProfitMetrics(Financials):
    CUSTOM_VARIABLES = {
        'current_position': [0, 2, np.int64],
        'trade_change': [-100000, 100000, np.float64],
        'running_profit': [-100000, 100000, np.float64]
    }
    PRICE_PAID = 0.0
    SET_PROFIT = 0.0

    def __init__(self, config: dict, pipeline: dict):
        super().__init__()

        warnings.filterwarnings('ignore', category=RuntimeWarning)

        self.logger = logging.getLogger(__name__)

        self.config = config
        self.pipeline = pipeline

        self.step = self.task_factory(self.config['task_selection'])


    def task_factory(self, task_selection: str) -> object:
        if task_selection == 'task3':
            return self.trading_step
        else:
            return self.state_step


    def initialise_variables(self) -> dict:
        custom_variables = {k: 0 for k in self.CUSTOM_VARIABLES.keys()}
        return custom_variables
    

    def reset_env_globals(self) -> None:
        self.PRICE_PAID = 0.0
        self.SET_PROFIT = 0.0
        return None
    

    def trading_step(self, payload: Payload, state_df: pd.DataFrame, custom_variable_dict: dict, terminated: bool) -> dict:
        if payload.update_state_data:
            action = payload.temp_action_int
            self.current_position = payload.action_dict[payload.previous_pos]
            self.strike_price = payload.last_price
        elif '_' in payload.active_pos:
            action = 0
            self.current_position = payload.action_dict[payload.previous_pos]
            self.strike_price = state_df['close'].iloc[-2]
        else:
            action = payload.action_int
            self.current_position = payload.action_dict[payload.active_pos]
            self.strike_price = state_df['close'].iloc[-2]

        self.strike_buy = self.strike_price
        self.strike_sell = self.strike_price

        custom_variable_dict = self.custom_variable_step(action, state_df, custom_variable_dict, terminated)

        return custom_variable_dict
    

    def state_step(self, action: int, state_df: pd.DataFrame, custom_variable_dict: dict, terminated: bool) -> dict:
        
        self.current_position = custom_variable_dict['current_position'][-1]

        self.strike_price = state_df['close'].iloc[-2]
        self.strike_buy = self.strike_price / self.price_penalty
        self.strike_sell = self.strike_price * self.price_penalty

        custom_variable_dict = self.custom_variable_step(action, state_df, custom_variable_dict, terminated)

        return custom_variable_dict
    

    def custom_variable_step(self, action: int, state_df: pd.DataFrame, custom_variable_dict: dict, terminated: bool) -> dict:
        self.determine_action_type(action, terminated)

        custom_variable_dict = self.current_position_update(action, custom_variable_dict)

        self.current_price_update(state_df)

        custom_variable_dict = self.trade_change_update(custom_variable_dict)

        custom_variable_dict = self.running_profit_update(custom_variable_dict)

        return custom_variable_dict
    

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

        self.logger.info(f'current position: {self.current_position}')
        self.logger.info(f'action type: {self.action_type}')
        
        return None
    

    def current_position_update(self, action: int, custom_variable_dict: dict) -> dict:

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

        custom_variable_dict['current_position'] = np.append(custom_variable_dict['current_position'], position_dict[self.action_type])

        self.logger.info(f'new current position: {position_dict[self.action_type]}')

        return custom_variable_dict
    

    def set_current_prices(self, state_df: pd.DataFrame) -> None:
        self.new_price = state_df['close'].iloc[-1]
        self.new_buy = self.new_price / self.price_penalty
        self.new_sell = self.new_price * self.price_penalty
        self.price_paid_buy = self.PRICE_PAID / self.price_penalty
        self.price_paid_sell = self.PRICE_PAID * self.price_penalty

        return None
    

    def current_price_update(self, state_df: pd.DataFrame) -> None:
        self.set_current_prices(state_df)

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

        self.logger.info(f'previous price: {self.strike_price}')
        self.logger.info(f'new price: {self.new_price}')
        self.logger.info(f'price paid: {self.PRICE_PAID}')

        return None
    
    
    def trade_change_update(self, custom_variable_dict: dict) -> dict:
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

        custom_variable_dict['trade_change'] = np.append(custom_variable_dict['trade_change'], price_dict[self.action_type])

        self.logger.info(f'trade change: {price_dict[self.action_type]}')

        return custom_variable_dict
    

    def running_profit_update(self, custom_variable_dict: dict) -> dict:
        session_profit = custom_variable_dict['running_profit'][-1]
        local_profit = self.SET_PROFIT
        trade_change = custom_variable_dict['trade_change'][-1]
        last_trade_change = custom_variable_dict['trade_change'][-2]

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

        custom_variable_dict['running_profit'] = np.append(custom_variable_dict['running_profit'], state_update)

        self.logger.info(f'running profit: {state_update}')

        return custom_variable_dict


    @strategy_wrapper_function
    def predict(self, state: dict) -> tuple:
        action = 0
        _states = state
        return action, _states