from datetime import datetime
from decimal import Decimal
import logging
import os
import pandas as pd
import pytz
from .stream_queue import StreamQueue

class StreamFaker:
    START_DATEPART = -4
    END_DATEPART = -8
    DATE_COLUMN = 'date'
    VALID_FORMAT = '%Y%m%d %H:%M:%S'
    VALID_POS = -11

    def __init__(self, config: dict, pipeline: dict):
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.pipeline = pipeline

        self.decimal_columns = self.pipeline['pipeline']['live_data_config']['fakerColumnTypes']
        self.timezone = self.pipeline['pipeline']['timezone']

        self.queue = StreamQueue(self.config, self.pipeline)


    def read_data(self) -> None:
        data_path = self.config['data_path']
        saved_data_path = os.path.join(data_path, 'saved_data/')
        pipeline_name = self.pipeline['pipeline']['filename']
        pipeline_data_path = os.path.join(saved_data_path, f'{pipeline_name}/')

        date_text = [f[:self.START_DATEPART][self.END_DATEPART:] for f in os.listdir(pipeline_data_path) if os.path.isfile(os.path.join(pipeline_data_path, f)) and f.endswith('.csv')]

        for date_str in date_text:
            try:
                # Try to convert the string to a date object
                date_obj = datetime.strptime(date_str, '%Y%m%d')
                file_date = date_obj
                break
            except ValueError:
                # If conversion fails, skip this element
                self.logger.error('No date string found in filename')
                continue

        file_trim = self.pipeline['pipeline']['model_data_config']['file_trim']

        contract_info = self.pipeline['pipeline']['contract_info']

        filename = '{}_{}_{}{:02d}{:02d}.csv'.format(
            contract_info['symbol'],
            contract_info['primaryExchange'],
            file_date.year,
            file_date.month,
            file_date.day
        )

        file_path = os.path.join(pipeline_data_path, filename)

        if os.path.exists(file_path):
            self.logger.info(f'Reading: {filename}')
        else:
            self.logger.info(f'Skipping: {filename} (file not found or not a CSV file)')
        
        try:
            # Read the current CSV file into a DataFrame
            self.final_dataframe = pd.read_csv(file_path)

            rows_to_drop = int(len(self.final_dataframe) * file_trim / 2)

            self.final_dataframe = self.final_dataframe.iloc[rows_to_drop:-rows_to_drop]

        except Exception as e:
            self.logger.info(f'Error reading {filename}: {e}')
            raise e

        self.episode_length = len(self.final_dataframe) - self.pipeline['pipeline']['model_data_config']['past_events']
        
        return None
    

    def process_time(self, dt_str) -> int:
        dt = datetime.strptime(dt_str[:self.VALID_POS], self.VALID_FORMAT)
        local_dt = pytz.timezone(self.timezone).localize(dt)
        epoch_time = local_dt.timestamp()
        
        return epoch_time
    

    def process_data(self, row: pd.Series) -> dict:
        bar = row.to_dict()

        bar[self.DATE_COLUMN] = int(bar[self.DATE_COLUMN])

        if self.decimal_columns:
            for col, dtype in self.decimal_columns.items():
                if dtype == 'int':
                    col_mod = int(bar[col])
                elif dtype == 'float':
                    col_mod = float(bar[col])
                else:
                    col_mod = bar[col]

                col_mod = str(col_mod)

                bar[col] = Decimal(col_mod)

        return bar


    def send_data(self) -> None:
        self.final_dataframe[self.DATE_COLUMN] = self.final_dataframe[self.DATE_COLUMN].apply(self.process_time)

        processed_data = self.final_dataframe.apply(self.process_data, axis=1)

        for index, row in processed_data.items():
            self.logger.info(f'Sending row: {index}')
            self.queue.put(row)

        return None
    
    
    def run(self) -> None:
        self.read_data()

        self.send_data()

        return None