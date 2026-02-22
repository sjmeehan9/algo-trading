"""Reinforcement learning training orchestration using the RLTrainer abstraction."""

import logging
import os

import pandas as pd
from algotrading.src.trainers import RLTrainer, TrainingConfig
from stable_baselines3.common.env_checker import check_env

from ..data_sourcing.state_builder import StateBuilder
from ..envs.trading_env import TradingEnv
from ..reward_functions.reward import reward_factory
from .trainer_factory import create_rl_trainer


class TrainRL:
    """Orchestrates RL model training and evaluation through an RLTrainer."""

    REWARD = "reward"
    ACTION = "action"

    def __init__(
        self,
        config: dict,
        pipeline: dict,
        path_dict: dict,
        evaluate: bool,
        trainer: RLTrainer | None = None,
    ):
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.pipeline = pipeline
        self.path_dict = path_dict
        self.evaluate = evaluate

        self.reward_name = self.pipeline["pipeline"]["model"]["model_reward"]
        self.env_name = self.pipeline["pipeline"]["env_config"]["env_name"]
        self.model_type = self.pipeline["pipeline"]["model"]["model_type"]
        self.model_config = self.pipeline["pipeline"]["model"]["model_config"]
        self.model_input = self.config["input_model"]
        self.model_filename = self.config["save_to_file"]
        self.trainer = (
            trainer if trainer is not None else create_rl_trainer(self.pipeline)
        )

    def write_session_info(self, session: dict) -> dict:
        training_dates = [d.isoformat() for d in self.state_builder.master_date_list]

        session_info = {
            "model": self.model_filename,
            "input_model": self.model_input,
            "reward_function": self.reward_name,
            "env": self.env_name,
            "training_dates": training_dates,
            "total_timesteps": self.state_builder.total_timesteps,
            "data_mode": self.config["data_mode"],
            "episode_length": self.state_builder.episode_length,
        }

        session.update(session_info)

        return session

    def data_setup(self) -> dict:
        if self.config["data_mode"] == "historical":
            self.state_builder.read_data(self.evaluate)
            self.logger.info("Data read from historical files")
            return None
        elif self.config["data_mode"] == "live":
            self.state_builder.live_data_function()
            raise NotImplementedError("Live data is not yet supported")
        else:
            self.logger.error("data_mode not recognised")
            return None

    def env_factory(self, env_name: str) -> object:
        if env_name == "trading_env":
            env = TradingEnv(self.state_builder)
            try:
                check_env(env)
                return env
            except Exception as e:
                self.logger.error(f"Environment check failed: {e}")
                raise e
        else:
            self.logger.error("env_name not recognised")
            return None

    def evaluate_factory(self, model_type: str) -> None:
        supported_model_types = {"ppo", "dqn"}
        if model_type not in supported_model_types:
            self.logger.error("model_type not recognised")
            raise ValueError("model_type not recognised")

        self.trainer.load(self.path_dict["eval_filepath"], self.env)
        self.logger.info(f'Loaded model from {self.path_dict["eval_filepath"]}')

        self.evaluate_model()

        return None

    def evaluate_model(self) -> None:
        while not self.state_builder.timed_out:
            state, info = self.env.reset()
            self.logger.info("Environment reset complete")

            self.eval_dataframe = self.state_builder.state_df.copy()

            self.eval_dataframe[self.REWARD] = 0.0
            self.eval_dataframe[self.ACTION] = 0

            # Add the custom variables to the dataframe
            for key in self.reward.CUSTOM_VARIABLES.keys():
                self.eval_dataframe[key] = state[key]

            # Loop through the environment
            while not self.state_builder.terminated:
                action, _states = self.trainer.predict(state)

                state, reward, terminated, truncated, info = self.env.step(action)

                # Get the index of the last row
                last_row_index = self.state_builder.state_df.index[-1]

                # Select and copy the last row
                last_row = self.state_builder.state_df.loc[[last_row_index]].copy()

                # Loop through each key and update the copy
                for key in self.reward.CUSTOM_VARIABLES.keys():
                    last_row[key] = state[key][-1]

                last_row[self.REWARD] = reward
                last_row[self.ACTION] = action

                # Append the modified last row to eval_dataframe
                self.eval_dataframe = pd.concat(
                    [self.eval_dataframe, last_row], ignore_index=True
                )

            self.logger.info("Episode terminated")

            # Save eval_dataframe to CSV file after the loop terminates
            eval_date = self.state_builder.master_date_list[
                self.state_builder.state_counters["window"]
            ].strftime("%Y%m%d")
            csv_file_name = f'{self.path_dict["pipeline_backtest_path"]}{eval_date}.csv'
            self.eval_dataframe.to_csv(csv_file_name, index=False)
            self.logger.info(f"Saved eval_dataframe to {csv_file_name}")

        return None

    def training_factory(self, model_type: str) -> object:
        supported_model_types = {"ppo", "dqn"}
        if model_type not in supported_model_types:
            self.logger.error("model_type not recognised")
            raise ValueError("model_type not recognised")

        return self.run_training()

    def _build_training_config(self) -> TrainingConfig:
        """Translate pipeline model config into a TrainingConfig dataclass."""
        custom_params = dict(self.model_config)
        learning_rate = custom_params.pop("learning_rate", None)
        batch_size = custom_params.pop("batch_size", None)
        n_steps = custom_params.pop("n_steps", None)

        return TrainingConfig(
            total_timesteps=self.state_builder.total_timesteps,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_steps=n_steps,
            tensorboard_log=self.path_dict["tensorboard_path"],
            custom_params=custom_params,
        )

    def run_training(self) -> None:
        """Execute trainer-delegated model creation, training, and persistence."""
        config = self._build_training_config()

        if os.path.exists(self.path_dict["input_filepath"]):
            self.trainer.load(self.path_dict["input_filepath"], self.env)
            self.logger.info(f'Loaded model from {self.path_dict["input_filepath"]}')
        else:
            self.trainer.create_model(self.env, config)
            self.logger.info(f"Created new {self.model_type.upper()} model")

        self.trainer.train(config)
        self.trainer.save(self.path_dict["model_filepath"])
        self.logger.info(f'Saved model to {self.path_dict["model_filepath"]}')

        return None

    def start(self) -> None:
        # Instanciate reward function object
        self.reward = reward_factory(self.reward_name, self.config, self.pipeline)

        # Instanciate StateBuilder object
        self.state_builder = StateBuilder(self.config, self.pipeline, self.reward)

        # Setup the data feed
        self.data_setup()

        # Contruct initial state dictionary
        self.state_builder.initialise_state()

        # Instanciate environment object
        self.env = self.env_factory(self.env_name)

        if self.evaluate:
            # Run backtest
            self.evaluate_factory(self.model_type)
        else:
            # Run training
            self.training_factory(self.model_type)

        return None
