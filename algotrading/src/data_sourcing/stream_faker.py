from datetime import datetime
import logging
import os
import pandas as pd

class StreamFaker:
    START_DATEPART = -4
    END_DATEPART = -8

    def __init__(self, config: dict, pipeline: dict, queue: object):
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.pipeline = pipeline

        self.queue = queue


    def read_data(self) -> None:
        data_path = self.config['data_path']
        saved_data_path = os.path.join(data_path, 'saved_data/')
        pipeline_name = self.pipeline['pipeline']['filename']
        pipeline_data_path = os.path.join(saved_data_path, f'{pipeline_name}/')

        date_text = [f[:self.START_DATEPART][self.END_DATEPART:] for f in os.listdir(pipeline_data_path) if os.path.isfile(os.path.join(pipeline_data_path, f)) and f.endswith('.csv')]

        date_list = []
        for date_str in date_text:
            try:
                # Try to convert the string to a date object
                date_obj = datetime.strptime(date_str, '%Y%m%d')
                date_list.append(date_obj)
            except ValueError:
                # If conversion fails, skip this element
                self.logger.error('No date string found in filename')
                continue

        file_trim = self.pipeline['pipeline']['model_data_config']['file_trim']

        contract_info = self.pipeline['pipeline']['contract_info']

        self.final_dataframe = pd.DataFrame()

        self.master_date_list = date_list.copy()

        date = date_list[0]

        filename = '{}_{}_{}{:02d}{:02d}.csv'.format(
            contract_info['symbol'],
            contract_info['primaryExchange'],
            date.year,
            date.month,
            date.day
        )

        file_path = os.path.join(pipeline_data_path, filename)

        if os.path.exists(file_path):
            self.logger.info(f'Reading: {filename}')
        else:
            self.master_date_list.remove(date)
            self.logger.info(f'Skipping: {filename} (file not found or not a CSV file)')
        
        try:
            # Read the current CSV file into a DataFrame
            current_df = pd.read_csv(file_path)

            rows_to_drop = int(len(current_df) * file_trim / 2)

            current_df = current_df.iloc[rows_to_drop:-rows_to_drop]
            
            # Append the current DataFrame to the final DataFrame
            self.final_dataframe = pd.concat([self.final_dataframe, current_df], ignore_index=True)

        except Exception as e:
            self.logger.info(f'Error reading {filename}: {e}')
            raise e

        self.episode_length = len(current_df) - self.pipeline['pipeline']['model_data_config']['past_events']
        
        return None


    def send_data(self) -> None:
        # Iterate over the final DataFrame
        for index, row in self.final_dataframe.iterrows():
            self.logger.info(f'Sending row: {index}')

            bar = row.to_dict()

            self.queue.put(bar)

        return None
    
    
    def start(self) -> None:
        self.read_data()

        self.send_data()

        return None