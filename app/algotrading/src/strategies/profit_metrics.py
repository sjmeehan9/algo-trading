import logging
import numpy as np
import warnings
from .strategy_wrapper import strategy_wrapper_function
from ..trading.financials import Financials

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


    @strategy_wrapper_function
    def predict(self, state: dict) -> int:
        return 0