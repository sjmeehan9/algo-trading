"""State management service extracted from legacy StateBuilder."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from algotrading.src.data_pipeline.state.scaler import ScalerWrapper
from algotrading.src.data_pipeline.types import DataBatch, DataRecord


@dataclass(slots=True)
class StateConfig:
    """Configuration for rolling state construction.

    Args:
        window_size: Number of past events per observation window.
        columns: Mapping of column name to (is_key, should_scale) flags.
        scaler_type: Scaler implementation name.
        trim_percentage: Reserved for compatibility; trimming occurs in source layer.
    """

    window_size: int
    columns: dict[str, tuple[bool, bool]]
    scaler_type: str = "MinMaxScaler"
    trim_percentage: float = 0.0


class StateManager:
    """Manage rolling state windows, scaling, and custom variable updates.

    Extracts state construction, windowing, scaling, and counter
    management from the legacy ``StateBuilder`` into a standalone
    service that operates on ``DataBatch`` or raw ``DataFrame``
    inputs.
    """

    def __init__(self, config: StateConfig, custom_logic: object | None = None) -> None:
        """Initialise the state manager.

        Args:
            config: Rolling window and scaling configuration.
            custom_logic: Optional strategy object providing
                ``initialise_variables`` and ``step`` methods for
                custom state variable injection.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.custom_logic = custom_logic

        self.final_dataframe = pd.DataFrame()
        self._state_df = pd.DataFrame()
        self._state: dict[str, np.ndarray] = {}

        self.episode_length = 0
        self.total_timesteps = 0
        self.file_offset = 0
        self.file_step = 0
        self.window_end = self.config.window_size

        self.custom_variables: dict[str, Any] = {}
        self.terminated = False
        self.timed_out = False
        self.initialised = False
        self._is_initialized = False

        self._configure_columns()
        self.scaler = ScalerWrapper(self.config.scaler_type)
        self.initialise_counters()

    def _configure_columns(self) -> None:
        self.key_columns: list[str] = []
        self.scale_columns: list[str] = []
        self.unscaled_columns: list[str] = []

        for column_name, options in self.config.columns.items():
            is_key, should_scale = options
            if is_key:
                self.key_columns.append(column_name)
            elif should_scale:
                self.scale_columns.append(column_name)
            else:
                self.unscaled_columns.append(column_name)

    def load_data(self, data: DataBatch) -> None:
        """Load batch data into internal dataframe state.

        Args:
            data: Homogeneous market/news batch to use as state source.
        """

        dataframe = data.to_dataframe()
        if "timestamp" in dataframe.columns and "date" not in dataframe.columns:
            dataframe = dataframe.rename(columns={"timestamp": "date"})
        self.load_dataframe(dataframe)

    def load_dataframe(self, df: pd.DataFrame) -> None:
        """Load a dataframe directly and reset internal state pointers."""

        self.final_dataframe = df.copy()
        self._state_df = pd.DataFrame()
        self._state = {}
        self._is_initialized = False

    def append_record(self, record: DataRecord) -> None:
        """Append a single streaming record to the internal dataframe."""

        row = {"date": record.timestamp}
        row.update(record.payload)
        self.final_dataframe = pd.concat(
            [self.final_dataframe, pd.DataFrame([row])], ignore_index=True
        )

    def initialise_state(self) -> dict[str, np.ndarray]:
        """Construct initial rolling state window and return state dictionary."""

        if self.custom_logic is not None:
            self.custom_variables = self.custom_logic.initialise_variables()
        else:
            self.custom_variables = {}

        frame_start = self.file_step
        frame_end = self.file_step + self.window_end
        window_df = self.final_dataframe.iloc[frame_start:frame_end]
        if window_df.empty:
            raise ValueError("Cannot initialise state from an empty data window")

        self._state_df = window_df
        self._state = self._build_state_dict(
            window_df=window_df,
            scaled=self._scale_columns(window_df),
            custom_vars=self.custom_variables,
        )
        self._is_initialized = True
        return self._state

    def state_step(self, action: int) -> tuple[dict[str, np.ndarray], bool]:
        """Advance one step and return the new state and termination flag."""

        self.state_counters["step"] += 1

        self.file_step = self.state_counters["step"] + self.file_offset

        frame_start = self.file_step
        frame_end = self.file_step + self.window_end
        window_df = self.final_dataframe.iloc[frame_start:frame_end]
        if window_df.empty:
            self.terminated = True
            return self._state, self.terminated

        self._state_df = window_df

        if self.state_counters["step"] == self.episode_length:
            self.terminated = True

        if (
            self.state_counters["step"] * self.state_counters["episode"]
            == self.total_timesteps
        ):
            self.timed_out = True

        custom_variable_dict = self._update_custom_variables(
            action=action,
            terminated=self.terminated,
        )

        self._state = self._build_state_dict(
            window_df=window_df,
            scaled=self._scale_columns(window_df),
            custom_vars=custom_variable_dict,
        )
        return self._state, self.terminated

    def reset(self) -> dict[str, np.ndarray]:
        """Reset state progression for a new episode and return fresh state."""

        self.state_counters["step"] = 0
        self.terminated = False
        return self.initialise_state()

    def get_current_window(self) -> pd.DataFrame:
        """Return a copy of the current observation dataframe window."""

        return self._state_df.copy()

    def initialise_counters(self) -> None:
        """Initialize step/window/episode counters."""

        self.state_counters = {"step": 0, "window": 0, "episode": 1}

    def update_episode_counter(self) -> None:
        """Increment episode/window counters for historical iteration."""

        self.state_counters["window"] += 1
        self.state_counters["episode"] += 1

    @property
    def current_step(self) -> int:
        """Current step index in the active episode."""

        return self.state_counters["step"]

    @property
    def current_episode(self) -> int:
        """Current episode number."""

        return self.state_counters["episode"]

    @property
    def state(self) -> dict[str, np.ndarray]:
        """Current state dictionary used by environments and trading logic."""

        return self._state

    @property
    def state_df(self) -> pd.DataFrame:
        """Current observation window dataframe."""

        return self._state_df

    @property
    def total_steps(self) -> int:
        """Total number of timesteps across all episodes."""

        return self.total_timesteps

    @property
    def is_initialized(self) -> bool:
        """Whether initial state has been built."""

        return self._is_initialized

    def set_custom_logic(self, logic: object) -> None:
        """Replace custom logic strategy used for custom state variables."""

        self.custom_logic = logic

    def _scale_columns(self, df: pd.DataFrame) -> np.ndarray:
        """Scale configured columns and return numpy matrix."""

        if not self.scale_columns:
            return np.empty((len(df), 0))

        scale_df = df[self.scale_columns]
        return self.scaler.fit_transform(scale_df.to_numpy())

    def _update_custom_variables(
        self, action: int, terminated: bool
    ) -> dict[str, np.ndarray]:
        """Compute custom variable values for the newest state window."""

        if self.custom_logic is None or not self.custom_variables:
            return {}

        previous_custom = {
            key: self._state[key][1:]
            for key in self.custom_variables.keys()
            if key in self._state
        }

        return self.custom_logic.step(
            action, self._state_df, previous_custom, terminated
        )

    def _build_state_dict(
        self,
        window_df: pd.DataFrame,
        scaled: np.ndarray,
        custom_vars: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Assemble final state dictionary with scaled/unscaled/custom columns."""

        temp_dataframe = pd.DataFrame()

        if self.scale_columns:
            scaled_df = pd.DataFrame(scaled, columns=self.scale_columns)
            temp_dataframe = pd.concat([temp_dataframe, scaled_df], axis=1)

        if self.unscaled_columns:
            unscaled_df = window_df[self.unscaled_columns].reset_index(drop=True)
            temp_dataframe = pd.concat([temp_dataframe, unscaled_df], axis=1)

        if custom_vars:
            for key, value in custom_vars.items():
                temp_dataframe[key] = value

        state = temp_dataframe.to_dict(orient="list")
        return {key: np.array(value) for key, value in state.items()}
