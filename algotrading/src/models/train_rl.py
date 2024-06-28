import logging
import os
import pandas as pd
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_checker import check_env
from ..data_sourcing.state_builder import StateBuilder
from ..envs.trading_env import TradingEnv
from ..reward_functions.reward import reward_factory

class TrainRL:
    REWARD = 'reward'
    ACTION = 'action'

    def __init__(self, config: dict, pipeline: dict, path_dict: dict, evaluate: bool):
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.pipeline = pipeline
        self.path_dict = path_dict
        self.evaluate = evaluate

        # Instanciate StateBuilder object
        self.state_builder = StateBuilder(self.config, self.pipeline)

        self.reward_name = self.pipeline['pipeline']['model']['model_reward']
        self.env_name = self.pipeline['pipeline']['env_config']['env_name']
        self.model_type = self.pipeline['pipeline']['model']['model_type']
        self.model_policy = self.pipeline['pipeline']['model']['model_policy']
        self.model_config = self.pipeline['pipeline']['model']['model_config']
        self.model_input = self.config['input_model']
        self.model_filename = self.config['save_to_file']


    def write_session_info(self, session: dict) -> dict:
        training_dates = [d.isoformat() for d in self.state_builder.master_date_list]

        session_info = {
            'model': self.model_filename,
            'input_model': self.model_input,
            'reward_function': self.reward_name,
            'env': self.env_name,
            'training_dates': training_dates,
            'total_timesteps': self.state_builder.total_timesteps,
            'data_mode': self.config['data_mode'],
            'episode_length': self.state_builder.episode_length
        }

        session.update(session_info)

        return session


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
        if env_name == 'trading_env':
            env = TradingEnv(self.state_builder)
            try:
                check_env(env)
                return env
            except Exception as e:
                self.logger.error(f'Environment check failed: {e}')
                raise e
        else:
            self.logger.error('env_name not recognised')
            return None
        
    
    def evaluate_factory(self, model_type: str) -> None:
        if model_type == 'ppo':
            self.model = PPO.load(self.path_dict['eval_filepath'], self.env)
            self.logger.info(f'Loaded PPO model from {self.path_dict["eval_filepath"]}')
        elif model_type == 'dqn':
            self.model = DQN.load(self.path_dict['eval_filepath'], self.env)
            self.logger.info(f'Loaded DQN model from {self.path_dict["eval_filepath"]}')
        else:
            self.logger.error('model_type not recognised')
        
        self.evaluate_model()
        
        return None
    

    def evaluate_model(self) -> None:
        while not self.state_builder.timed_out:
            state, info = self.env.reset()
            self.logger.info('Environment reset complete')

            self.eval_dataframe = self.state_builder.state_df.copy()

            self.eval_dataframe[self.REWARD] = 0.0
            self.eval_dataframe[self.ACTION] = 0

            # Add the custom variables to the dataframe
            for key in self.reward.CUSTOM_VARIABLES.keys():
                self.eval_dataframe[key] = state[key]

            # Loop through the environment
            while not self.state_builder.terminated:
                action, _states = self.model.predict(state)

                state, reward, terminated, truncated, info = self.env.step(action)

                # Get the index of the last row
                last_row_index = self.state_builder.state_df.index[-1]

                # Select and copy the last row
                last_row = self.state_builder.state_df.loc[[last_row_index]].copy()

                # Loop through each key and update the copy
                for key in self.reward.CUSTOM_VARIABLES.keys():
                    last_row[key] = state[key][-1]

                last_row[self.REWARD] = reward
                last_row[self.ACTION] = action.item()

                # Append the modified last row to eval_dataframe
                self.eval_dataframe = pd.concat([self.eval_dataframe, last_row], ignore_index=True)
            
            self.logger.info('Episode terminated')

            # Save eval_dataframe to CSV file after the loop terminates
            eval_date = self.state_builder.master_date_list[self.state_builder.state_counters['window']].strftime("%Y%m%d")
            csv_file_name = f'{self.path_dict["pipeline_backtest_path"]}{eval_date}.csv'
            self.eval_dataframe.to_csv(csv_file_name, index=False)
            self.logger.info(f'Saved eval_dataframe to {csv_file_name}')
        
        return None
        

    def training_factory(self, model_type: str) -> object:
        if model_type == 'ppo':
            return self.train_ppo()
        elif model_type == 'dqn':
            return self.train_dqn()
        else:
            self.logger.error('model_type not recognised')
            return None
    

    def train_dqn(self) -> None:
        # Create input arguments for DQN model
        model_kwargs = {
            'policy': self.model_policy,
            'env': self.env,
            'tensorboard_log': self.path_dict['tensorboard_path']
        }

        model_kwargs.update(self.model_config)

        # Load input model or create new model
        if os.path.exists(self.path_dict['input_filepath']):
            self.model = DQN.load(self.path_dict['input_filepath'], self.env)
            self.logger.info(f'Loaded DQN model from {self.path_dict["input_filepath"]}')
        else:
            self.model = DQN(**model_kwargs)
            self.logger.info('Created new DQN model')
        
        # Train the model
        self.model.learn(total_timesteps=self.state_builder.total_timesteps, tb_log_name=self.model_filename)

        # Save the model
        self.model.save(self.path_dict['model_filepath'])

        return None
    

    def train_ppo(self) -> None:
        # Create input arguments for DQN model
        model_kwargs = {
            'policy': self.model_policy,
            'env': self.env,
            'tensorboard_log': self.path_dict['tensorboard_path']
        }

        model_kwargs.update(self.model_config)

        # Load input model or create new model
        if os.path.exists(self.path_dict['input_filepath']):
            self.model = PPO.load(self.path_dict['input_filepath'], self.env)
            self.logger.info(f'Loaded PPO model from {self.path_dict["input_filepath"]}')
        else:
            self.model = PPO(**model_kwargs)
            self.logger.info('Created new PPO model')
        
        # Train the model
        self.model.learn(total_timesteps=self.state_builder.total_timesteps, tb_log_name=self.model_filename)

        # Save the model
        self.model.save(self.path_dict['model_filepath'])

        return None


    def start(self) -> None:
        # Setup the data feed
        self.data_setup()

        # Instanciate reward function object
        self.reward = reward_factory(self.reward_name, self.config, self.pipeline)

        # Contruct initial state dictionary
        self.state_builder.initialise_state(self.reward)

        # Instanciate environment object
        self.env = self.env_factory(self.env_name)

        if self.evaluate:
            # Run backtest
            self.evaluate_factory(self.model_type)
        else:
            # Run training
            self.training_factory(self.model_type)

        return None
