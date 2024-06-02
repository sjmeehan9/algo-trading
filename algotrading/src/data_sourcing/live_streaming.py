import datetime
from datetime import timedelta
import logging
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import os
from pathlib import Path
import pytz
from ..load_config import config_loader
from .stream_queue import StreamQueue

    
class LiveData(EWrapper, EClient):
    CONFIG_FILENAME = 'live_streaming.yml'
    HISTORICAL_CONFIG = 'historical_data.yml'
    CURRENT_BAR = ''
    BASE_SECONDS = 20
    INIT_REQUEST_ID = 1000
    TIMEZONE = 'US/Eastern'
    DATE_COLUMN = 'date'
    VALID_FORMAT = '%Y%m%d %H:%M:%S'
    VALID_POS = -11

    def __init__(self, config: dict, pipeline: dict):
        EClient.__init__(self, self)

        self.logger = logging.getLogger(__name__)
        
        self.config = config
        self.pipeline = pipeline

        self.queue = StreamQueue(self.config, self.pipeline)

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
                self.historical_info['barSizeSetting'], self.historical_info['whatToShow'], 1, 2, False, [])
        

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

        self.CURRENT_BAR = ''

        self.data_validation = self.dataValidation()

        if self.data_validation:
            self.logger.info('Data validation passed')
        else:
            self.logger.error('Data validation failed, dropping data list')
            self.data_list = []

        self.logger.info(f'Full data list: {self.data_list}')

        #TEMP
        self.realtimeBar(self.req_it, 0,0,0,0,0,0,0,0)

        # self.reqRealTimeBars(self.req_it, self.contract, self.live_info['barSizeSetting'], 
        #             self.live_info['whatToShow'], True, [])
        
    
    def dataValidation(self) -> bool:
        dates = [datetime.datetime.fromtimestamp(int(d[self.DATE_COLUMN])) for d in self.data_list]

        # Determine the time increment in seconds
        self.time_increment = int(self.live_info['barSizeSetting'])

        # Calculate the expected number of entries. Only going to work for seconds
        start_date = dates[0]
        self.end_date = dates[-1]
        expected_count = int((self.end_date - start_date).total_seconds() / self.time_increment) + 1

        # Check if the actual count matches the expected count
        if len(dates) != expected_count:
            return False

        # Check if all expected datetimes are present
        current_date = start_date
        for date in dates:
            if date != current_date:
                return False
            current_date += timedelta(seconds=self.time_increment)

        return True
            

    def reformatTime(self, time: int) -> str:
        dt_utc = datetime.datetime.fromtimestamp(time, pytz.utc)
        dt_eastern = dt_utc.astimezone(pytz.timezone(self.TIMEZONE))
        formatted_date = dt_eastern.strftime(f'{self.VALID_FORMAT} {self.TIMEZONE}')
        
        return formatted_date
    

    def connectDates(self, time_string: str) -> bool:
        time = datetime.datetime.strptime(time_string[:self.VALID_POS], self.VALID_FORMAT)
        connect_time = time - timedelta(seconds=self.time_increment)

        if connect_time == self.end_date:
            self.logger.info('Final historical bar increments to real time data, transferring to real time stream')
            return True
        elif connect_time > self.end_date:
            self.logger.info('Gap between historical data end time exists, dropping data')
            self.data_list = []
            return True
        else:
            self.logger.info('Real time data overlaps with historical end time, waiting for new data')
            return False

    
    # Receive live data
    def realtimeBar(self, reqId, time, open_, high, low, close, volume, wap, count) -> None:

        from decimal import Decimal
        
        time = 1717185595
        open_ = 165.47
        high = 165.55
        low = 165.45
        close = 165.49
        volume = Decimal('24012')
        wap = 165.47
        count = Decimal('165.4974508579044')
        
        #TODO:
        # Truncate to only required columns?
        # Log

        time_string = self.reformatTime(time)

        new_row = {self.bar_columns['bar_date']: time_string, 
                    self.bar_columns['bar_open']: open_, 
                    self.bar_columns['bar_high']: high, 
                    self.bar_columns['bar_low']: low, 
                    self.bar_columns['bar_close']: close, 
                    self.bar_columns['bar_volume']: volume, 
                    self.bar_columns['bar_wap']: wap, 
                    self.bar_columns['bar_barCount']: count}

        if not self.CURRENT_BAR:
            process = self.connectDates(time_string)
            if process:
                self.data_list.append(new_row)
                self.queue.put(new_row)
                self.CURRENT_BAR = time

        elif self.CURRENT_BAR != time:
            self.queue.put(new_row)

            self.CURRENT_BAR = time

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
