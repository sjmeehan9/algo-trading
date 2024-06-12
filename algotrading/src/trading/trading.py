import logging
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ..models.predict import Predict
from .order import OrderManager

class Trading(EWrapper, EClient):
    BASE_SECONDS = 20

    def __init__(self, config: dict, pipeline: dict):
        EClient.__init__(self, self)

        self.logger = logging.getLogger(__name__)

        self.config = config
        self.pipeline = pipeline

        self.predict = Predict(self.config, self.pipeline)
        self.order = OrderManager(self.config, self.pipeline)

        self.timer = self.setTimer()


    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson) -> None:
        self.logger.info(f'Error: {reqId}, {errorCode}, {errorString}')


    def connect(self, ip_address, port, client_id):
        super().connect(ip_address, port, client_id)


    def nextValidId(self, orderId: int) -> None:
        super().nextValidId(orderId)
        self.nextValidOrderId = orderId

    
    def setTimer(self) -> int:
        return self.BASE_SECONDS
        

    def trading_algorithm(self, state: dict) -> None:
        action, _ = self.predict.get_action(state)

        action_int = action.item()

        self.logger.info(f'action: {action_int}')

        return action_int
    
    
    def stop(self) -> None:
        self.logger.info('Trading connection closed')
        
        self.done = True
        self.disconnect()