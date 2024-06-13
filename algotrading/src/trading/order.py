import logging
from ibapi.order import Order

class OrderManager:
    def __init__(self, config: dict, pipeline: dict):
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.pipeline = pipeline


    def checkAction(self, action, active_pos) -> bool:
        if action != 'NONE' and action != active_pos and '_PEND' not in active_pos and '_FILL' not in active_pos:
            return True
        else:
            return False
        

    def calcOrderSpec(self, balance, units, action, activePos, price):
        
        #TODO Move multiplier to pipeline config
        adj_balance = balance * 0.9
        unit_amt = round(adj_balance / price, 0)
        mod_string = activePos + action
        
        if units == 0:
            units = unit_amt
        
        self.logger.info(f'Figures: adj_balance {adj_balance}, unit_amt {unit_amt}, mod_string {mod_string}')
        
        action_dict = {'NONEBUY': ['BUY', unit_amt],
                       'SELLBUY': ['NONE', abs(units)],
                       'NONESELL': ['SELL', unit_amt],
                       'BUYSELL': ['NONE', abs(units)]}
        
        self.logger.info(f'Order spec: {action_dict[mod_string]}')
        
        return action_dict[mod_string]


    def executeOrder(self, orderAction, units) -> tuple:
        
        order = Order()
        order.action = orderAction
        order.totalQuantity = units
        order.orderType = 'MKT'
        order.eTradeOnly = False
        order.firmQuoteOnly = False
        
        self.logger.info(f'units: {units}')
        
        #TODO No init_oType in self
        if active_pos == 'NONE' and orderAction not in self.init_oType:
            pass
        else:
            active_pos = '{}_PEND'.format(orderAction)

        return order, active_pos


    def positionUnlock(self, active_position, balance_figure, balance_list, update_position) -> tuple:
        
        balance_list.append(balance_figure)
        
        if '_FILL' in active_position and len(balance_list) == 2:
            return_pos = update_position
            balance_list = []
            
        else:
            return_pos = active_position
            
        self.logger.info(f'{return_pos}, {balance_list}')
        
        return return_pos, balance_list