import logging
import os
from datetime import datetime
from typing import Any, Callable

import pandas as pd

from ..data_pipeline.state.state_manager import StateConfig, StateManager
from ..trading.trading import Trading


class StateBuilder:
    """Backward-compatible façade over ``StateManager``.

    Retains legacy file-loading and task-routing behaviour while
    delegating rolling-window state construction, scaling, and
    counter management to ``StateManager``.
    """

    START_DATEPART = -4
    END_DATEPART = -8

    def __init__(self, config: dict, pipeline: dict, custom_logic: Any):
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.pipeline = pipeline
        self.custom_logic = custom_logic

        state_config = StateConfig(
            window_size=self.pipeline["pipeline"]["state_data_config"]["past_events"],
            columns=self.pipeline["pipeline"]["state_data_config"]["columns"],
            scaler_type=self.pipeline["pipeline"]["state_data_config"].get(
                "scaler", "MinMaxScaler"
            ),
        )
        self.state_manager = StateManager(
            config=state_config, custom_logic=custom_logic
        )

        self.initialise_counters()

        self.terminated = False
        self.timed_out = False
        self.file_offset = 0
        self.file_step = 0

        self.live_data_function = self.initialise_live_data()

    def read_data(self, evaluate: bool) -> None:
        """Load CSV files for training or evaluation and sync to StateManager."""
        data_path = self.config["data_path"]
        saved_data_path = os.path.join(data_path, "saved_data/")
        pipeline_name = self.pipeline["pipeline"]["filename"]
        pipeline_data_path = os.path.join(saved_data_path, f"{pipeline_name}/")

        if evaluate:
            date_list = self.config["backtest_date_list"]
        else:
            date_list = self.config["training_date_list"]

        # If date_list is empty, add all filenames in the pipeline_data_path to date_list
        if not date_list:
            date_text = [
                f[: self.START_DATEPART][self.END_DATEPART :]
                for f in os.listdir(pipeline_data_path)
                if os.path.isfile(os.path.join(pipeline_data_path, f))
                and f.endswith(".csv")
            ]

            date_list = []
            for date_str in date_text:
                try:
                    # Try to convert the string to a date object
                    date_obj = datetime.strptime(date_str, "%Y%m%d")
                    date_list.append(date_obj)
                except ValueError:
                    # If conversion fails, skip this element
                    self.logger.error("No date string found in filename")
                    continue

        file_trim = self.pipeline["pipeline"]["state_data_config"]["file_trim"]

        contract_info = self.pipeline["pipeline"]["contract_info"]

        self.final_dataframe = pd.DataFrame()
        current_df = pd.DataFrame()

        self.master_date_list = date_list.copy()

        for date in date_list:
            filename = "{}_{}_{}{:02d}{:02d}.csv".format(
                contract_info["symbol"],
                contract_info["primaryExchange"],
                date.year,
                date.month,
                date.day,
            )

            file_path = os.path.join(pipeline_data_path, filename)

            if os.path.exists(file_path):
                self.logger.info(f"Reading: {filename}")
            else:
                self.master_date_list.remove(date)
                self.logger.info(
                    f"Skipping: {filename} (file not found or not a CSV file)"
                )
                continue

            try:
                # Read the current CSV file into a DataFrame
                current_df = pd.read_csv(file_path)

                rows_to_drop = int(len(current_df) * file_trim / 2)

                current_df = current_df.iloc[rows_to_drop:-rows_to_drop]

                # Append the current DataFrame to the final DataFrame
                self.final_dataframe = pd.concat(
                    [self.final_dataframe, current_df], ignore_index=True
                )

            except Exception as e:
                self.logger.info(f"Error reading {filename}: {e}")
                continue

        self.episode_length = (
            len(current_df)
            - self.pipeline["pipeline"]["state_data_config"]["past_events"]
        )

        self.total_timesteps = len(self.master_date_list) * self.episode_length

        if self.pipeline["pipeline"]["state_data_config"]["columns"]:
            columns_keys = list(
                self.pipeline["pipeline"]["state_data_config"]["columns"].keys()
            )
            self.final_dataframe = self.final_dataframe[columns_keys]

        if not isinstance(self.final_dataframe, pd.DataFrame):
            raise TypeError("StateBuilder final_dataframe must be a pandas DataFrame")

        self.state_manager.load_dataframe(self.final_dataframe)
        self.state_manager.episode_length = self.episode_length
        self.state_manager.total_timesteps = self.total_timesteps
        self.state_manager.state_counters = self.state_counters
        self.state_manager.terminated = self.terminated
        self.state_manager.timed_out = self.timed_out

        return None

    def load_dataframe(
        self,
        dataframe: pd.DataFrame,
        *,
        episode_length: int | None = None,
        total_timesteps: int | None = None,
    ) -> None:
        """Load pre-normalized market data for API-backed training workflows."""

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("StateBuilder dataframe input must be a pandas DataFrame")
        if dataframe.empty:
            raise ValueError("StateBuilder dataframe input must not be empty")

        state_config = self.pipeline["pipeline"]["state_data_config"]
        required_columns = list(state_config["columns"].keys())
        missing_columns = [
            column for column in required_columns if column not in dataframe
        ]
        if missing_columns:
            raise ValueError(
                "StateBuilder dataframe is missing required columns: "
                + ", ".join(missing_columns)
            )

        window_size = int(state_config["past_events"])
        if len(dataframe) <= window_size:
            raise ValueError(
                "StateBuilder dataframe must contain more rows than past_events"
            )

        resolved_episode_length = (
            int(episode_length)
            if episode_length is not None
            else len(dataframe) - window_size
        )
        if resolved_episode_length < 1:
            raise ValueError("episode_length must be positive")

        resolved_total_timesteps = (
            int(total_timesteps)
            if total_timesteps is not None
            else resolved_episode_length
        )
        if resolved_total_timesteps < 1:
            raise ValueError("total_timesteps must be positive")

        self.initialise_counters()
        self.terminated = False
        self.timed_out = False
        self.file_offset = 0
        self.file_step = 0
        self.final_dataframe = dataframe[required_columns].reset_index(drop=True).copy()
        self.episode_length = resolved_episode_length
        self.total_timesteps = resolved_total_timesteps
        self.master_date_list = self._dataframe_dates(self.final_dataframe)

        self.state_manager.load_dataframe(self.final_dataframe)
        self.state_manager.episode_length = self.episode_length
        self.state_manager.total_timesteps = self.total_timesteps
        self.state_manager.state_counters = self.state_counters
        self.state_manager.terminated = self.terminated
        self.state_manager.timed_out = self.timed_out

        return None

    def _dataframe_dates(self, dataframe: pd.DataFrame) -> list[datetime]:
        parsed = pd.to_datetime(dataframe["date"], utc=True, errors="coerce")
        dates = sorted({item.date() for item in parsed.dropna()})
        return [datetime.combine(item, datetime.min.time()) for item in dates]

    def initialise_counters(self) -> None:
        """Reset step, window, and episode counters."""
        # Setup counters
        self.state_counters = {"step": 0, "window": 0, "episode": 1}

        self.state_manager.state_counters = self.state_counters

        return None

    def task_state(self) -> None:
        """Configure file offset and dataframe based on active task."""
        if self.config["task_selection"] == "task3":
            self.file_offset = 0
            self.final_dataframe = self.queue
        elif (
            self.config["task_selection"] == "task2"
            and self.config["data_mode"] == "live"
        ):
            self.file_offset = 0
            self.final_dataframe = self.queue
        elif (
            self.config["task_selection"] == "task2"
            and self.config["data_mode"] == "historical"
        ):
            self.terminated = False
            self.timed_out = False
            self.file_offset = self.state_counters["window"] * (
                self.episode_length + self.window_end
            )
        elif self.config["task_selection"] == "task4":
            self.terminated = False
            self.timed_out = False
            self.file_offset = self.state_counters["window"] * (
                self.episode_length + self.window_end
            )
        else:
            self.logger.error("Data usage not supported")
            raise NotImplementedError("Data usage not supported")

        self.state_manager.file_offset = self.file_offset
        self.state_manager.file_step = self.file_offset + self.state_counters["step"]
        self.state_manager.terminated = self.terminated
        self.state_manager.timed_out = self.timed_out

        return None

    def initialise_state(self) -> None:
        """Build initial observation state via StateManager delegation."""
        self.window_end = self.pipeline["pipeline"]["state_data_config"]["past_events"]

        self.task_state()

        self.state_manager.window_end = self.window_end
        self.state_manager.file_offset = self.file_offset
        self.state_manager.file_step = self.state_counters["step"] + self.file_offset
        self.state_manager.custom_logic = self.custom_logic

        if not isinstance(self.final_dataframe, pd.DataFrame):
            raise TypeError("StateBuilder final_dataframe must be a pandas DataFrame")

        self.state_manager.load_dataframe(self.final_dataframe)
        self.state = self.state_manager.initialise_state()
        self.state_df = self.state_manager.state_df
        self.custom_variables = self.state_manager.custom_variables
        self.key_columns = self.state_manager.key_columns
        self.scale_columns = self.state_manager.scale_columns
        self.unscaled_columns = self.state_manager.unscaled_columns
        self.scaler = self.state_manager.scaler

        return None

    def state_step(self, action: int) -> None:
        """Advance state one step, delegate to StateManager."""
        self.file_offset = self.state_counters["window"] * (
            self.episode_length + self.window_end
        )
        self.state_manager.file_offset = self.file_offset
        self.state_manager.file_step = self.state_counters["step"] + self.file_offset
        self.state_manager.total_timesteps = self.total_timesteps
        self.state_manager.episode_length = self.episode_length
        self.state_manager.terminated = self.terminated
        self.state_manager.timed_out = self.timed_out

        self.state, self.terminated = self.state_manager.state_step(action)
        self.timed_out = self.state_manager.timed_out
        self.state_df = self.state_manager.state_df

        self.logger.info(f'first frame date: {self.state_df["date"].iloc[0]}')
        self.logger.info(f'last frame date: {self.state_df["date"].iloc[-1]}')

        return None

    def update_episode_counter(self) -> None:
        """Increment window and episode counters for historical iteration."""
        self.state_counters["window"] += 1
        self.state_counters["episode"] += 1
        return None

    def initialise_live_data(self) -> Callable[[], None] | None:
        """Route live data handling to the correct task function."""
        self.logger.info(f"Initialising live data function")

        route: Callable[[], None] | None = None

        if (
            self.config["task_selection"] == "task2"
            or self.config["task_selection"] == "task3"
        ):
            # Flag for completed initialisation
            self.initialised = False

            # Route to the correct function based on the task selection
            if self.config["task_selection"] == "task2":
                route = self.live_step
            elif self.config["task_selection"] == "task3":
                self.terminated = False

                app = Trading(self.config, self.pipeline)
                self.trading = app

                route = self.trading_step

            self.logger.info(f"Initialised live data function")
        else:
            self.logger.error("Live data usage not supported or used for this task")
            route = None

        return route

    def live_data(self, queue: pd.DataFrame) -> None:
        """Accept incoming live data and trigger state update."""
        self.logger.info(f"Live data received by StateBuilder")
        self.queue = queue

        if self.initialised:
            if self.live_data_function is not None:
                self.live_data_function()
        else:
            self.initialise_state()
            self.initialised = True

        return None

    def trading_step(self) -> None:
        """Rebuild state for live trading and execute trade logic."""
        self.final_dataframe = self.queue

        self.state_manager.load_dataframe(self.final_dataframe)
        self.state_manager.file_offset = self.file_offset
        self.state_manager.file_step = self.file_step
        self.state_manager.total_timesteps = self.total_timesteps
        self.state_manager.episode_length = self.episode_length
        self.state_manager.terminated = self.terminated
        self.state_manager.timed_out = self.timed_out

        # Rebuild state window for live trading without advancing counters.
        existing_custom_variables = dict(getattr(self, "custom_variables", {}))
        self.state = self.state_manager.initialise_state()
        self.state_df = self.state_manager.state_df
        if existing_custom_variables:
            self.custom_variables = existing_custom_variables
        else:
            self.custom_variables = self.state_manager.custom_variables

        payload = self.trading.payload

        custom_variable_dict = {
            key: self.state[key][1:]
            for key, _ in self.custom_variables.items()
            if key in self.state
        }
        custom_variable_dict = self.custom_logic.step(
            payload,
            self.state_df,
            custom_variable_dict,
            self.terminated,
        )

        self.state.update(custom_variable_dict)

        self.logger.info(f"state updated: {self.state}")

        self.trading.confirmTrades()

        # Sent state to trading_algorithm
        self.trading.tradingAlgorithm(self.state, self.state_df)

        return None

    def live_step(self) -> None:
        """Placeholder for task-2 live data stepping."""
        self.logger.info(f"Live step not yet implemented")
        return None

    @property
    def current_step(self) -> int:
        return self.state_manager.current_step

    @property
    def current_episode(self) -> int:
        return self.state_manager.current_episode

    @property
    def is_initialized(self) -> bool:
        return self.state_manager.is_initialized
