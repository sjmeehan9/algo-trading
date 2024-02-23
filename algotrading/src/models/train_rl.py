from ibapi.client import EClient
from ibapi.wrapper import EWrapper
import logging

class TrainRL(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)

        self.logger = logging.getLogger(__name__)

        
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson):
        self.logger.info(f'Error: {reqId}, {errorCode}, {errorString}')


    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.nextValidOrderId = orderId

        self.start()


    def start(self):
        pass
        
            
    def stop(self):
        self.done = True
        self.disconnect()
