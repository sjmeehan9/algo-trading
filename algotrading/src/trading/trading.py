import logging
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import pandas as pd
from threading import Timer
from ..models.predict import Predict
from .order import OrderManager
from .payload import Payload

class Trading(EWrapper, EClient):
    BASE_SECONDS = 60
    CURRENT_POS_LIST = []
    ACTIONS = {0: 'NONE', 1: 'BUY', 2: 'SELL', 'NONE': 0, 'BUY': 1, 'SELL': 2}

    def __init__(self, config: dict, pipeline: dict):
        EClient.__init__(self, self)

        self.logger = logging.getLogger(__name__)

        self.config = config
        self.pipeline = pipeline

        self.predict = Predict(self.config, self.pipeline)
        self.order = OrderManager(self.config, self.pipeline)
        self.payload = Payload()
        self.payload.action_dict = self.ACTIONS

        contract_info = self.pipeline['pipeline']['contract_info']
        self.contract = Contract()
        self.contract.symbol = contract_info['symbol']
        self.contract.secType = contract_info['secType']
        self.contract.exchange = contract_info['exchange']
        self.contract.currency = contract_info['currency']
        self.contract.primaryExchange = contract_info['primaryExchange']

        self.account = self.config['account_number']

        self.timer = self.setTimer()


    def error(self, reqId, errorCode, errorString) -> None:
        self.logger.info(f'Error: {reqId}, {errorCode}, {errorString}')


    def connect(self, ip_address, port, client_id):
        super().connect(ip_address, port, client_id)

        #
        self.logger.info('Trading connection request sent')

        self.start()


    def nextValidId(self, orderId: int) -> None:
        super().nextValidId(orderId)
        self.nextValidOrderId = orderId

        #
        self.logger.info('Starting Trading connection')


    def nextOrderId(self):
        self.oid = self.nextValidOrderId
        self.nextValidOrderId += 1
        return self.oid

    
    def setTimer(self) -> int:
        return self.BASE_SECONDS
    

    # Account data updates
    def updateAccountValue(self, key: str, val: str, currency: str, accountName: str) -> None:
        self.logger.info(f'Account update: {key} {val} {currency} {accountName}')

        if key == 'CashBalance' and currency == self.contract.currency:
            self.payload.cashbalance = float(val)

            self.logger.info(f'cashbalance: {self.payload.cashbalance}')

            if '_FILL' in self.payload.active_pos:
                self.payload.update_reward_vars, self.payload.current_pos_list = self.order.positionUnlock(self.payload.active_pos, self.payload.cashbalance, self.payload.current_pos_list, self.payload.order_spec[0])

    
    def updatePortfolio(self, contract: Contract, position: float, marketPrice: float, marketValue: float, averageCost: float, unrealizedPNL: float, realizedPNL: float, accountName: str) -> None:
        self.logger.info(f'Portfolio update: {contract} {position} {marketPrice} {marketValue} {averageCost} {unrealizedPNL} {realizedPNL} {accountName}')

        if contract.symbol == self.contract.symbol:
            self.payload.openunits = position
            
            self.logger.info(f'openunits: {self.payload.openunits}')
            
            if '_FILL' in self.payload.active_pos:
                self.payload.update_reward_vars, self.payload.current_pos_list = self.order.positionUnlock(self.payload.active_pos, self.payload.openunits, self.payload.current_pos_list, self.payload.order_spec[0])     

    
    def updateAccountTime(self, timeStamp: str) -> None:
        self.logger.info(f'Account time update: {timeStamp}')


    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        self.logger.info(f'OrderStatus. Id: {orderId}, Status: {status}, {filled}, {remaining}, {avgFillPrice}, {permId}, {parentId}, {lastFillPrice}, {clientId}, {whyHeld}, {mktCapPrice}')
        
        if (status == 'PreSubmitted' or status == 'Submitted') and filled == 0:
            self.timing = Timer(15, self.stopCancel, args=[self.oid])
            self.timing.start()
            self.timer = True
            
        elif (status == 'PreSubmitted' or status == 'Submitted') and filled > 0:
            self.payload.last_price = avgFillPrice
            self.payload.last_pos = self.payload.order_spec[0]
            self.payload.active_pos = '{}_PART'.format(self.payload.temp_action)
            
            self.logger.info(f'PART, {self.payload.active_pos}, {self.payload.last_pos}, {self.payload.last_price}')
            
        elif status == 'Filled':
            self.payload.active_pos = '{}_FILL'.format(self.payload.temp_action)
            self.timing.cancel()
            self.timer = False
            
            self.payload.last_price = avgFillPrice
            self.payload.last_pos = self.payload.order_spec[0]
            
            self.logger.info(f'FILL, {self.payload.active_pos}, {self.payload.last_pos}, {self.payload.last_price}')  


    def stopCancel(self, orderId):
        if '_PEND' in self.payload.active_pos or '_PART' in self.payload.active_pos:
            self.cancelOrder(orderId)
            self.timer = False

            if '_PEND' in self.payload.active_pos:
                self.payload.active_pos = self.payload.last_pos

            if '_PART' in self.payload.active_pos:
                self.payload.active_pos = '{}_FILL'.format(self.payload.temp_action)
        

    def trading_algorithm(self, state: dict) -> None:
        action, _ = self.predict.get_action(state)

        self.payload.action_int = action.item()

        self.payload.action_str = self.payload.action_dict[self.payload.action_int]

        self.logger.info(f'action: {self.payload.action_str}, model prediction: {self.payload.action_int}')

        if self.payload.release_trade == True:
            self.payload.active_pos = self.payload.last_pos
            self.payload.release_trade = False

        take_action = self.order.checkAction(self.payload.action_str, self.payload.active_pos)

        if take_action:
            self.logger.info('Action requested')
            self.executeOrder()
        else:
            self.logger.info('No action taken')

        return None
    

    def executeOrder(self) -> None:
        self.payload.previous_pos = self.payload.active_pos

        self.payload.temp_action = self.payload.action_str

        self.payload.temp_action_int = self.payload.action_int

        self.payload.order_spec = self.order.calcOrderSpec(self.payload.cashbalance, self.payload.openunits, self.payload.temp_action, self.payload.active_pos, self.payload.live_price)

        order, self.payload.active_pos = self.order.buildOrder(self.payload.temp_action, self.payload.order_spec[1])
        
        self.placeOrder(self.nextOrderId(), self.contract, order)

        return None
    

    def start(self) -> None:

        #
        self.logger.info('Calling reqAccountUpdates')
        
        self.reqAccountUpdates(True, '')
    
    
    def stop(self) -> None:
        self.logger.info('Trading connection closed')

        self.reqAccountUpdates(False, '')
        
        self.done = True
        self.disconnect()