import logging
import os
from pathlib import Path
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ..load_config import config_loader

    
class LiveData(EWrapper, EClient):
    CONFIG_FILENAME = 'live_streaming.yml'
    CURRENT_BAR = ''
    BASE_SECONDS = 20
    INIT_REQUEST_ID = 1000

    def __init__(self, config: dict, pipeline: dict):
        EClient.__init__(self, self)

        self.logger = logging.getLogger(__name__)
        
        self.config = config
        self.pipeline = pipeline

        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = Path(current_dir).parents[1]

        config_file_path = os.path.join(parent_dir, 'config/', self.CONFIG_FILENAME)
        self.script_config = config_loader(config_file_path)

        contract_info = self.pipeline['pipeline']['contract_info']
        self.contract = Contract()
        self.contract.symbol = contract_info['symbol']
        self.contract.secType = contract_info['secType']
        self.contract.exchange = contract_info['exchange']
        self.contract.currency = contract_info['currency']
        self.contract.primaryExchange = contract_info['primaryExchange']

        self.live_info = self.pipeline['pipeline']['live_data_config']

        self.bar_columns = self.script_config['bar_columns']

        self.timer = self.setTimer()

        
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson) -> None:
        self.logger.info(f'Error: {reqId}, {errorCode}, {errorString}')


    def nextValidId(self, orderId: int) -> None:
        super().nextValidId(orderId)
        self.nextValidOrderId = orderId

        self.start()

    
    def setTimer(self) -> int:
        return self.BASE_SECONDS
    

    def sendRequests(self) -> None:

        self.reqRealTimeBars(self.req_it, self.contract, self.live_info['barSizeSetting'], 
                            self.live_info['whatToShow'], True, [])
        
    
    # Receive historical data
    def realtimeBar(self, reqId, time, open_, high, low, close, volume, wap, count) -> None:

        if not self.CURRENT_BAR:
            self.CURRENT_BAR = time

        elif self.CURRENT_BAR != time:
            new_row = {self.bar_columns['bar_date']: time, 
                       self.bar_columns['bar_open']: open_, 
                       self.bar_columns['bar_high']: high, 
                       self.bar_columns['bar_low']: low, 
                       self.bar_columns['bar_close']: close, 
                       self.bar_columns['bar_volume']: volume, 
                       self.bar_columns['bar_wap']: wap, 
                       self.bar_columns['bar_barCount']: count}
            
            self.CURRENT_BAR = time

            #TODO: Do something with the data
            self.logger.info(f'New row data: {new_row}')


    def start(self) -> None:       

        logging.getLogger().setLevel(logging.INFO)
        self.req_it = self.INIT_REQUEST_ID

        # Request live realTimeBars data
        self.sendRequests()

            
    def stop(self) -> None:

        self.cancelRealTimeBars(self.req_it)
        
        self.done = True
        self.disconnect()
