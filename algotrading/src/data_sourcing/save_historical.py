import datetime
import logging
import os
import pandas as pd
from pathlib import Path
import time
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ..load_config import config_loader

class PastData(EWrapper, EClient):
    CONFIG_FILENAME = 'historical_data.yml'
    CURRENT_BAR = ''
    BASE_SECONDS = 3
    INIT_REQUEST_ID = 1000
    SLEEP_DURATION = 1
    LOAD_DURATION = 10
    DATE_STR_POS = 8
    DATE_COLUMN = 'date'

    def __init__(self, config: dict, pipeline: dict):
        EClient.__init__(self, self)

        self.logger = logging.getLogger(__name__)
        
        self.config = config
        self.pipeline = pipeline
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = Path(current_dir).parents[1]

        config_file_path = os.path.join(parent_dir, 'config/', self.CONFIG_FILENAME)
        self.script_config = config_loader(config_file_path)
        
        data_path = self.config['data_path']
        saved_data_path = os.path.join(data_path, 'saved_data/')
        pipeline_name = self.pipeline['pipeline']['filename']
        pipeline_data_path = os.path.join(saved_data_path, f'{pipeline_name}/')

        # Create log and data folders
        if not os.path.exists(saved_data_path):
            os.makedirs(saved_data_path)

        if not os.path.exists(pipeline_data_path):
            os.makedirs(pipeline_data_path)

        self.folder_name = pipeline_data_path

        contract_info = self.pipeline['pipeline']['contract_info']
        self.contract = Contract()
        self.contract.symbol = contract_info['symbol']
        self.contract.secType = contract_info['secType']
        self.contract.exchange = contract_info['exchange']
        self.contract.currency = contract_info['currency']
        self.contract.primaryExchange = contract_info['primaryExchange']

        self.bar_columns = self.script_config['bar_columns']

        self.timezone = self.pipeline['pipeline']['timezone']

        self.historical_info = self.pipeline['pipeline']['historical_data_config']
        self.data_df = pd.DataFrame(columns=self.historical_info['columns'])
        self.data_list = []

        self.step_size = self.script_config['step_size'][self.historical_info['barSizeSetting']]

        self.max_loops = self.step_size['loops_required']

        self.first_date = self.config['date_list'][0]
        self.date_list = self.config['date_list'][1:]

        self.timer = self.setTimer()

        
    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson) -> None:
        self.logger.info(f'Error: {reqId}, {errorCode}, {errorString}')


    def nextValidId(self, orderId: int) -> None:
        super().nextValidId(orderId)
        self.nextValidOrderId = orderId

        self.start()

    
    def setTimer(self) -> int:
        num_dates = len(self.config['date_list'])
        loops_required = self.max_loops

        return self.BASE_SECONDS * num_dates * loops_required
    

    def checkDataframe(self, date_requested: str) -> bool:
        dt_min = self.data_df[self.DATE_COLUMN].min()[:self.DATE_STR_POS]
        dt_max = self.data_df[self.DATE_COLUMN].max()[:self.DATE_STR_POS]
        row_check = (self.step_size['durationNum'] / self.step_size['barSize']) * self.step_size['loops_required']

        if dt_min == dt_max and self.data_df.shape[0] == row_check and dt_min == date_requested:
            return True
        else:
            return False


    def adjustTime(self, time, minutes) -> datetime.datetime:
        return time - datetime.timedelta(minutes=minutes)
    

    def sendRequests(self, date: datetime.date) -> None:
        self.end_date = datetime.datetime.combine(date, datetime.time(
            self.step_size['date_hour_max'],
            self.step_size['date_minute_max'],
            self.step_size['date_second_max']
        ))

        self.loop_it = 0
        temp_time = self.end_date

        while self.loop_it < self.step_size['loops_required']:
            adj_time = self.adjustTime(temp_time, self.step_size['increment_size'])

            temp_time_str = '{}{:02d}{:02d} {:02d}:{:02d}:{:02d} {}'.format(
                temp_time.year, temp_time.month, temp_time.day, temp_time.hour, temp_time.minute, 
                temp_time.second, self.timezone)

            self.reqHistoricalData(self.req_it, self.contract, temp_time_str, self.step_size['durationString'], 
                self.historical_info['barSizeSetting'], self.historical_info['whatToShow'], 1, 1, False, [])

            self.loop_it += 1
            self.req_it += 1
            temp_time = adj_time
            time.sleep(self.SLEEP_DURATION)
        
    
    # Receive historical data
    def historicalData(self, reqId, bar) -> None:

        if not self.CURRENT_BAR:
            self.CURRENT_BAR = bar.date

        elif self.CURRENT_BAR != bar.date:
            new_row = {self.bar_columns['bar_date']: bar.date, 
                       self.bar_columns['bar_open']: bar.open, 
                       self.bar_columns['bar_high']: bar.high, 
                       self.bar_columns['bar_low']: bar.low, 
                       self.bar_columns['bar_close']: bar.close, 
                       self.bar_columns['bar_volume']: bar.volume, 
                       self.bar_columns['bar_wap']: bar.wap, 
                       self.bar_columns['bar_barCount']: bar.barCount}
            self.data_list.append(new_row)
            self.CURRENT_BAR = bar.date


    def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:

        self.cancelHistoricalData(reqId)

        # After reqId cycles through the loops required, save data to csv
        iter_num = (reqId - self.INIT_REQUEST_ID + 1) % self.step_size['loops_required']
        date_requested = end[:self.DATE_STR_POS]

        if iter_num == 0:
            self.data_df = pd.DataFrame(self.data_list)
            self.data_df = self.data_df.drop_duplicates(subset=self.DATE_COLUMN)
            self.data_df = self.data_df[self.historical_info['columns']]
            self.data_df = self.data_df.fillna(0)
            self.data_df = self.data_df.sort_values(self.DATE_COLUMN)

            if self.checkDataframe(date_requested):
                self.data_df.to_csv('{}/{}_{}_{}.csv'.format(self.folder_name, self.contract.symbol, 
                    self.contract.primaryExchange, date_requested), index=False)

            else:
                self.logger.info('Error: Date mismatch or incorrect row count in dataframe')

            time.sleep(self.LOAD_DURATION)
            
            self.data_list = []
            
            # Loop through the remaining dates
            if self.date_list:
                self.sendRequests(self.date_list.pop(0))
            else:
                self.stop()


    def start(self) -> None:       

        self.req_it = self.INIT_REQUEST_ID

        # Request historical data for first date in the list
        self.sendRequests(self.first_date)

            
    def stop(self) -> None:
        
        self.done = True
        self.disconnect()
