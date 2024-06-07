from datetime import datetime
import logging
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ..reward_functions.profit_seeker import ProfitSeeker

class StateBuilder:
    START_DATEPART = -4
    END_DATEPART = -8
    SCALER = MinMaxScaler()

    def __init__(self, config: dict, pipeline: dict):
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.pipeline = pipeline

        self.initialise_counters()

        self.live_data_function = self.initialise_live_data()


    def read_data(self, evaluate: bool) -> None:
        data_path = self.config['data_path']
        saved_data_path = os.path.join(data_path, 'saved_data/')
        pipeline_name = self.pipeline['pipeline']['filename']
        pipeline_data_path = os.path.join(saved_data_path, f'{pipeline_name}/')

        if evaluate:
            date_list = self.config['backtest_date_list']
        else:
            date_list = self.config['training_date_list']

        # If date_list is empty, add all filenames in the pipeline_data_path to date_list
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

        self.master_date_list = date_list.copy()

        for date in date_list:
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

        self.total_timesteps = len(self.master_date_list) * self.episode_length
        
        if self.pipeline['pipeline']['model_data_config']['columns']:
            columns_keys = list(self.pipeline['pipeline']['model_data_config']['columns'].keys())
            self.final_dataframe = self.final_dataframe[columns_keys]
        
        return None
    

    def initialise_counters(self) -> None:
        # Setup counters
        self.state_counters = {
            'step': 0,
            'window': 0,
            'episode': 1
        }

        return None


    def initialise_state(self, reward: object) -> None:
        self.terminated = False

        self.timed_out = False

        self.reward = reward

        self.reward_variables = self.reward.initial_reward_variables()

        self.window_end = self.pipeline['pipeline']['model_data_config']['past_events']

        self.file_offset = self.state_counters['window'] * (self.episode_length + self.window_end)

        self.file_step = self.state_counters['step'] + self.file_offset

        # Grab the first window of data
        frame_start = self.file_step
        frame_end = self.file_step + self.window_end
        self.state_df = self.final_dataframe.iloc[frame_start:frame_end]

        # Loop through model data config columns and save three lists 
        # A key list, a scale list and an unscaled list
        self.key_columns = []
        self.scale_columns = []
        self.unscaled_columns = []

        for key, value in self.pipeline['pipeline']['model_data_config']['columns'].items():
            if value[0]==True:
                self.key_columns.append(key)
            elif value[1]==True:
                self.scale_columns.append(key)
            else:
                self.unscaled_columns.append(key)

        # Create an empty dataframe
        temp_dataframe = pd.DataFrame()
        
        # If scale_columns list not empty, append to a temp dataframe
        if self.scale_columns:
            temp_dataframe = self.state_df[self.scale_columns]

            scaled_data = self.SCALER.fit_transform(temp_dataframe)

            temp_dataframe = pd.DataFrame(scaled_data, columns=self.scale_columns)

        # If unscale_columns list not empty, append to temp dataframe
        if self.unscaled_columns:
            unscaled_df = self.state_df[self.unscaled_columns].reset_index(drop=True)

            temp_dataframe = pd.concat([temp_dataframe, unscaled_df], axis=1)

        # Add the default custom reward values
        if self.reward_variables:
            for key, value in self.reward_variables.items():
                temp_dataframe[key] = value

        # store the state dictionary
        self.state = temp_dataframe.to_dict(orient='list')

        self.state = {key: np.array(value) for key, value in self.state.items()}

        return None

    
    def state_step(self, action: int) -> None:
        self.state_counters['step'] += 1

        self.file_offset = self.state_counters['window'] * (self.episode_length + self.window_end)

        self.file_step = self.state_counters['step'] + self.file_offset

        # Grab the next window of data
        frame_start = self.file_step
        frame_end = self.file_step + self.window_end
        self.state_df = self.final_dataframe.iloc[frame_start:frame_end]

        self.logger.info(f'first frame date: {self.state_df["date"].iloc[0]}')
        self.logger.info(f'last frame date: {self.state_df["date"].iloc[-1]}')

        # Create an empty dataframe
        temp_dataframe = pd.DataFrame()
        
        # If scale_columns list not empty, append to a temp dataframe
        if self.scale_columns:
            temp_dataframe = self.state_df[self.scale_columns]

            scaled_data = self.SCALER.fit_transform(temp_dataframe)

            temp_dataframe = pd.DataFrame(scaled_data, columns=self.scale_columns)

        # If unscale_columns list not empty, append to temp dataframe
        if self.unscaled_columns:
            unscaled_df = self.state_df[self.unscaled_columns].reset_index(drop=True)

            temp_dataframe = pd.concat([temp_dataframe, unscaled_df], axis=1)
        
        # For the custom variables, grab the previous step values from the last step
        reward_variable_dict = {key: self.state[key][1:] for key, value in self.reward_variables.items()}

        # Check if the episode is over
        if self.state_counters['step'] == self.episode_length:
            self.terminated = True

        # Check if total_timesteps has been reached
        if self.state_counters['step'] * self.state_counters['episode'] == self.total_timesteps:
            self.timed_out = True

        # Call each reward variable function to calculate the new values for the latest time
        reward_variable_dict = self.reward.reward_variable_step(action, self.state_df, reward_variable_dict, self.terminated)

        # store the state dictionary
        self.state = temp_dataframe.to_dict(orient='list')

        self.state = {key: np.array(value) for key, value in self.state.items()}

        self.state.update(reward_variable_dict)
        
        return None


    def update_episode_counter(self) -> None:
        self.state_counters['window'] += 1
        self.state_counters['episode'] += 1
        return None

    
    def initialise_live_data(self) -> object:
        # Init reward for use in live data tasks
        reward_name = self.pipeline['pipeline']['model']['model_reward']
        if reward_name == 'profit_seeker':
            self.reward = ProfitSeeker(self.config, self.pipeline)
        else:
            self.logger.error('reward_name not recognised')
            return None

        # Route to the correct function based on the task selection
        if self.config['task_selection'] == 'task2':
            route = self.live_step
        elif self.config['task_selection'] == 'task3':
            route = self.trading_step
        else:
            self.logger.error('Live data usage not supported')
            route = None
        
        return route


    def live_data(self, queue: pd.DataFrame) -> None:
        self.logger.info(f'Live data received by StateBuilder')

        self.live_data_function()

        return None
    

    def trading_step(self) -> None:
        self.logger.info(f'Trading step')
        return None


    def live_step(self) -> None:
        self.logger.info(f'Live step')
        return None
