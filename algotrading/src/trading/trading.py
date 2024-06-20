import logging
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from threading import Timer
from ..models.predict import Predict
from .order import OrderManager
from .payload import Payload

class Trading(EWrapper, EClient):
    BASE_SECONDS = 60
    CURRENT_POS_LIST = []

    def __init__(self, config: dict, pipeline: dict):
        EClient.__init__(self, self)

        self.logger = logging.getLogger(__name__)

        self.config = config
        self.pipeline = pipeline

        self.predict = Predict(self.config, self.pipeline)
        self.order = OrderManager(self.config, self.pipeline)
        self.payload = Payload()

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
            self.cashbalance = float(val)

            self.logger.info(f'cashbalance: {self.cashbalance}')

            if '_FILL' in self.active_pos:
                self.active_pos, self.current_pos_list = self.order.positionUnlock(self.active_pos, self.cashbalance, self.CURRENT_POS_LIST, self.order_spec[0])

    
    def updatePortfolio(self, contract: Contract, position: float, marketPrice: float, marketValue: float, averageCost: float, unrealizedPNL: float, realizedPNL: float, accountName: str) -> None:
        self.logger.info(f'Portfolio update: {contract} {position} {marketPrice} {marketValue} {averageCost} {unrealizedPNL} {realizedPNL} {accountName}')

        if contract.symbol == self.contract.symbol:
            self.openunits = position
            
            self.logger.info(f'openunits: {self.openunits}')
            
            if '_FILL' in self.active_pos:
                self.active_pos, self.current_pos_list = self.order.positionUnlock(self.active_pos, self.openunits, self.CURRENT_POS_LIST, self.order_spec[0])     

    
    def updateAccountTime(self, timeStamp: str) -> None:
        self.logger.info(f'Account time update: {timeStamp}')


    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        self.logger.info(f'OrderStatus. Id: {orderId}, Status: {status}, {filled}, {remaining}, {avgFillPrice}, {permId}, {parentId}, {lastFillPrice}, {clientId}, {whyHeld}, {mktCapPrice}')
        
        if (status == 'PreSubmitted' or status == 'Submitted') and filled == 0:
            self.timing = Timer(15, self.stopCancel, args=[self.oid])
            self.timing.start()
            self.timer = True
            
        elif (status == 'PreSubmitted' or status == 'Submitted') and filled > 0:
            
            self.last_price = avgFillPrice
            self.last_pos = self.order_spec[0]
            self.active_pos = '{}_PEND'.format(self.temp_action)
            
            self.logger.info('PART, {self.active_pos}, {self.last_pos}, {self.last_price}')            
            
        elif status == 'Filled':
            self.timing.cancel()
            self.timer = False
            
            self.last_price = avgFillPrice
            self.last_pos = self.order_spec[0]
            self.active_pos = '{}_FILL'.format(self.temp_action)
            
            self.logger.info('FILL, {self.active_pos}, {self.last_pos}, {self.last_price}')  


    def stopCancel(self, orderId):
        if '_PEND' in self.active_pos:
            self.cancelOrder(orderId)
            self.timer = False
            self.active_pos = self.last_pos
        

    def trading_algorithm(self, state: dict) -> None:
        action, _ = self.predict.get_action(state)

        action_int = action.item()

        self.logger.info(f'action: {action_int}')

        take_action = self.order.checkAction(action_int, self.active_pos)

        if take_action:
            #TODO Does temp action need to be parsed along?
            self.temp_action = action_int

            #TODO No last_price yet
            self.order_spec = self.order.calcOrderSpec(self.cashbalance, self.openunits, action_int, self.active_pos, 
                                                       self.last_price)

            #TODO Possibly execute within Trading class
            order, self.active_pos = self.order.executeOrder(self.contract, self.temp_action, self.order_spec[1])
            
            #TODO This should be a separate method along with above
            self.placeOrder(self.nextOrderId(), self.contract, order)
        else:
            self.logger.info('No action taken')

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