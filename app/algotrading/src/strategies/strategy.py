import logging
import pandas as pd
from stable_baselines3.common.env_checker import check_env
from .custom_logic import custom_logic_factory
from ..data_sourcing.state_builder import StateBuilder
from ..envs.strategy_env import StrategyEnv

class Strategy:
    ACTION = 'action'

    def __init__(self, config: dict, pipeline: dict, path_dict: dict = {}, evaluate: bool = False):
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.pipeline = pipeline
        self.path_dict = path_dict
        self.evaluate = evaluate

        self.env_name = self.pipeline['pipeline']['env_config']['env_name']
        self.strategy_name = self.pipeline['pipeline']['strategy']['strategy_name']


    def data_setup(self) -> dict:
        if self.config['data_mode'] == 'historical':
            self.state_builder.read_data(self.evaluate)
            self.logger.info('Data read from historical files')
            return None
        elif self.config['data_mode'] == 'live':
            self.state_builder.live_data_function()
            raise NotImplementedError('Live data is not yet supported')
        else:
            self.logger.error('data_mode not recognised')
            return None


    def env_factory(self, env_name: str) -> object:
        if env_name == 'strategy_env':
            env = StrategyEnv(self.state_builder)
            try:
                check_env(env)
                return env
            except Exception as e:
                self.logger.error(f'Environment check failed: {e}')
                raise e
        else:
            self.logger.error('env_name not recognised')
            return None


    def evaluate_strategy(self) -> None:
        while not self.state_builder.timed_out:
            state, info = self.env.reset()
            self.logger.info('Environment reset complete')

            self.eval_dataframe = self.state_builder.state_df.copy()

            self.eval_dataframe[self.ACTION] = 0

            # Add the custom variables to the dataframe
            for key in self.custom_logic.CUSTOM_VARIABLES.keys():
                self.eval_dataframe[key] = state[key]

            # Loop through the environment
            while not self.state_builder.terminated:
                action, _states = self.custom_logic.predict(state)
                action = action.item()

                state, reward, terminated, truncated, info = self.env.step(action)

                # Get the index of the last row
                last_row_index = self.state_builder.state_df.index[-1]

                # Select and copy the last row
                last_row = self.state_builder.state_df.loc[[last_row_index]].copy()

                # Loop through each key and update the copy
                for key in self.custom_logic.CUSTOM_VARIABLES.keys():
                    last_row[key] = state[key][-1]

                last_row[self.ACTION] = action

                # Append the modified last row to eval_dataframe
                self.eval_dataframe = pd.concat([self.eval_dataframe, last_row], ignore_index=True)
            
            self.logger.info('Episode terminated')

            # Save eval_dataframe to CSV file after the loop terminates
            eval_date = self.state_builder.master_date_list[self.state_builder.state_counters['window']].strftime("%Y%m%d")
            csv_file_name = f'{self.path_dict["pipeline_backtest_path"]}{eval_date}.csv'
            self.eval_dataframe.to_csv(csv_file_name, index=False)
            self.logger.info(f'Saved eval_dataframe to {csv_file_name}')
        
        return None


    def start(self) -> None:
        # Instanciate custom logic function object
        self.custom_logic = custom_logic_factory(self.strategy_name, self.config, self.pipeline)

        # Instanciate StateBuilder object
        self.state_builder = StateBuilder(self.config, self.pipeline, self.custom_logic)

        # Setup the data feed
        self.data_setup()

        # Contruct initial state dictionary
        self.state_builder.initialise_state()

        # Instanciate environment object
        self.env = self.env_factory(self.env_name)

        # Run backtest
        self.evaluate_strategy()

        return None