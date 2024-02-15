from ibapi.client import EClient
from ibapi.wrapper import EWrapper

class StateBuilder(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)

        
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson):
        print("Error: ", reqId, " ", errorCode, " ", errorString)


    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.nextValidOrderId = orderId

        self.start()


    def start(self):
        pass
        
            
    def stop(self):
        self.done = True
        self.disconnect()
