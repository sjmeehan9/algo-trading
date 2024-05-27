import logging
import os
from pathlib import Path
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ..load_config import config_loader

    
class LiveData(EWrapper, EClient):
    CONFIG_FILENAME = 'live_streaming.yml'
    HISTORICAL_CONFIG = 'historical_data.yml'
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

        historical_config_path = os.path.join(parent_dir, 'config/', self.HISTORICAL_CONFIG)
        self.historical_config = config_loader(historical_config_path)

        contract_info = self.pipeline['pipeline']['contract_info']
        self.contract = Contract()
        self.contract.symbol = contract_info['symbol']
        self.contract.secType = contract_info['secType']
        self.contract.exchange = contract_info['exchange']
        self.contract.currency = contract_info['currency']
        self.contract.primaryExchange = contract_info['primaryExchange']

        self.live_info = self.pipeline['pipeline']['live_data_config']
        self.historical_info = self.pipeline['pipeline']['historical_data_config']

        self.bar_columns = self.script_config['bar_columns']
        self.historical_columns = self.historical_config['bar_columns']

        self.step_size = self.historical_config['step_size'][self.historical_info['barSizeSetting']]
        self.data_list = []

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

        self.reqHistoricalData(self.req_it, self.contract, '', self.step_size['durationString'], 
                self.historical_info['barSizeSetting'], self.historical_info['whatToShow'], 1, 1, False, [])
        

    # Receive historical data
    def historicalData(self, reqId, bar) -> None:

        if not self.CURRENT_BAR:
            self.CURRENT_BAR = bar.date

        elif self.CURRENT_BAR != bar.date:
            new_row = {self.historical_columns['bar_date']: bar.date, 
                       self.historical_columns['bar_open']: bar.open, 
                       self.historical_columns['bar_high']: bar.high, 
                       self.historical_columns['bar_low']: bar.low, 
                       self.historical_columns['bar_close']: bar.close, 
                       self.historical_columns['bar_volume']: bar.volume, 
                       self.historical_columns['bar_wap']: bar.wap, 
                       self.historical_columns['bar_barCount']: bar.barCount}
            self.data_list.append(new_row)
            self.CURRENT_BAR = bar.date


    def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:

        self.cancelHistoricalData(reqId)

        self.req_it += 1

        self.reqRealTimeBars(self.req_it, self.contract, self.live_info['barSizeSetting'], 
                    self.live_info['whatToShow'], True, [])

    
    # Receive live data
    def realtimeBar(self, reqId, time, open_, high, low, close, volume, wap, count) -> None:

        #TODO: Wait until a new date is being received, if there's overlap in data received
        # If there's a gap, discard the historical data
        # Check by seeing if the last current date is 5s diff with the received time
        # Do above in a new function
        # If legit, add to date list and send to queue
        # Log

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
