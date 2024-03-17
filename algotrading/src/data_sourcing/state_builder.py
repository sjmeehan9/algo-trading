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

        # In order to set up obs space, we need to loop through the model columns
        # as well as the custom reward variables and understand what are keys
        # and what values need to be scaled


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
            columns_keys = list(self.pipeline['pipeline']['model_data_config']['columns'].keys())
            self.final_dataframe = self.final_dataframe[columns_keys]
        
        return None
    

    def initialise_state(self) -> dict:
        # Setup counters
        # Loop through model data config columns and save three lists 
        # A key list, a scale list and an unscaled list
        # If list not empty, append to a temp dataframe
        # Scale the scale columns as needed using a scale function, via an apply function
        # Create custom reward variables dictionary as class attribute
        # For now the dictionary element values are descriptions, keys are the variable names
        # Call the initial_reward_variables function to calculate the initial values
        # Return the state dictionary
        pass

    
    def state_step(self) -> dict:
        # Increment the counters
        # Using the lists from the initialise_state function, grab the next window of data
        # If list not empty, append to a temp dataframe
        # Scale the scale columns as needed using a scale function, via an apply function
        # For the custom variables, grab the previous step values from the last step
        # Drop the earliest row and add columns to dataframe
        # Call each reward variable function to calculate the new values for the latest time
        # Leverage calcs from the financials module
        # Return the state dictionary
        pass
