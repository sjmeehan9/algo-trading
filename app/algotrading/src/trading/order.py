import logging
from ibapi.order import Order

class OrderManager:
    def __init__(self, config: dict, pipeline: dict):
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.pipeline = pipeline

        self.order_type = self.pipeline['pipeline']['trading_config']['order_type']
        self.balance_multiplier = self.pipeline['pipeline']['trading_config']['balance_multiplier']


    def checkAction(self, action, active_pos) -> bool:
        self.logger.info(f'Action: {action}, Active position: {active_pos}')
        if action != 'NONE' and action != active_pos and '_PEND' not in active_pos and '_PART' not in active_pos and '_FILL' not in active_pos:
            return True
        else:
            return False
        

    def calcOrderSpec(self, balance, units, action, activePos, price) -> list:
        adj_balance = balance * self.balance_multiplier
        unit_amt = round(adj_balance / price, 0)
        mod_string = activePos + action
        
        if units == 0:
            units = unit_amt
        
        self.logger.info(f'Figures: adj_balance {adj_balance}, unit_amt {unit_amt}, mod_string {mod_string}')
        
        action_dict = {'NONEBUY': ['BUY', unit_amt, 'open'],
                       'SELLBUY': ['NONE', abs(units), 'close'],
                       'NONESELL': ['SELL', unit_amt, 'open'],
                       'BUYSELL': ['NONE', abs(units), 'close']}
        
        self.logger.info(f'Order spec: {action_dict[mod_string]}')
        
        return action_dict[mod_string]


    def buildOrder(self, orderAction, units) -> tuple:
        
        order = Order()
        order.action = orderAction
        order.totalQuantity = units
        order.orderType = self.order_type
        order.eTradeOnly = False
        order.firmQuoteOnly = False
        
        self.logger.info(f'trade units: {units}')
        
        active_pos = '{}_PEND'.format(orderAction)

        return order, active_pos


    def positionUnlock(self, active_position, balance_figure, balance_list, update_position) -> tuple:
        
        balance_list.append(balance_figure)
        
        if '_FILL' in active_position and len(balance_list) == 2:
            update_reward_vars = True
            balance_list = []
            
        else:
            update_reward_vars = False
            
        self.logger.info(f'{update_reward_vars}, {balance_list}')
        
        return update_reward_vars, balance_list