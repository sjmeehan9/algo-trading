from datetime import datetime
import logging
import os
import pandas as pd

class StateBuilder:
    START_DATEPART = -4
    END_DATEPART = -8

    def __init__(self, config: dict, pipeline: dict):
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.pipeline = pipeline

        # Build function to setup the state for step in the environment
        # Returns the window of data for the next step
        # Adds the custom reward variables to the state
        # Tracking of what place in the data we are is kept in StateBuilder


    def live_data(self):
        pass


    def read_data(self) -> None:
        data_path = self.config['data_path']
        saved_data_path = os.path.join(data_path, 'saved_data/')
        pipeline_name = self.pipeline['pipeline']['filename']
        pipeline_data_path = os.path.join(saved_data_path, f'{pipeline_name}/')

        date_list = self.config['training_date_list']

        # If training_date_list is empty, add all filenames in the pipeline_data_path to date_list
        if not date_list:
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

        for date in date_list:
            filename = '{}_{}_{}{:02d}{:02d}.csv'.format(
                contract_info['symbol'],
                contract_info['primaryExchange'],
                date.year,
                date.month,
                date.day
            )

            file_path = os.path.join(pipeline_data_path, filename)

            if not os.path.exists(file_path):
                self.logger.info(f'Skipping: {filename} (file not found or not a CSV file)')
                continue
            
            try:
                # Read the current CSV file into a DataFrame
                current_df = pd.read_csv(file_path)

                rows_to_drop = int(len(current_df) * file_trim / 2)

                current_df = current_df.iloc[rows_to_drop:-rows_to_drop]
                
                # Append the current DataFrame to the final DataFrame
                self.final_dataframe = pd.concat([self.final_dataframe, current_df], ignore_index=True)

            except Exception as e:
                self.logger.info(f'Error reading {filename}: {e}')
                continue

        self.episode_length = len(current_df) - self.pipeline['pipeline']['model_data_config']['past_events']
        
        if self.pipeline['pipeline']['model_data_config']['columns']:
            self.final_dataframe = self.final_dataframe[self.pipeline['pipeline']['model_data_config']['columns']]
        
        return None
