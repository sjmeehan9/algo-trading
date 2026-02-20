"""Shared base environment for trading and strategy evaluation."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from algotrading.src.trading.tools import TradingTools
from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete


class BaseTradingEnv(Env, ABC):
    """Base Gymnasium environment with shared trading logic."""

    DEFAULT_SPACE_MIN = 0
    DEFAULT_SPACE_MAX = 1000000
    ACTION_SPACE_SIZE = 3
    DUMMY_REWARD = 0

    def __init__(self, state_builder: object) -> None:
        super().__init__()

        self.logger = logging.getLogger(__name__)

        self.state_builder = state_builder

        # Actions we can take
        self.action_space = Discrete(self.ACTION_SPACE_SIZE)

        # Observation space
        self.observation_space = self._create_obs_space()

        # Trading rules that impact the environment
        self.tools = TradingTools(self.state_builder.pipeline)

    def _create_obs_space(self) -> Dict:
        """Create the observation space based on state builder configuration."""

        space_dict: dict[str, Box] = {}

        custom_variables = self.state_builder.custom_logic.CUSTOM_VARIABLES

        # Create the obs space from the StateBuilder data
        for key, value in self.state_builder.pipeline["pipeline"]["state_data_config"][
            "columns"
        ].items():
            if value[0] is True:
                continue
            if value[1] is True:
                space_dict[key] = Box(
                    low=min(self.state_builder.state[key]),
                    high=max(self.state_builder.state[key]),
                    shape=self.state_builder.state[key].shape,
                    dtype=np.float64,
                )
            else:
                space_dict[key] = Box(
                    low=self.DEFAULT_SPACE_MIN,
                    high=self.DEFAULT_SPACE_MAX,
                    shape=self.state_builder.state[key].shape,
                    dtype=np.float64,
                )

        if custom_variables:
            for key, value in custom_variables.items():
                space_dict[key] = Box(
                    low=value[0],
                    high=value[1],
                    shape=self.state_builder.state[key].shape,
                    dtype=value[2],
                )

        return Dict(space_dict)

    def step(
        self, action: int
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Execute a single environment step."""

        self.logger.info("step taken")

        action = self.tools.stop_take(action, self.state_builder.state)

        self.state_builder.state_step(action)

        reward = self._get_reward(action, self.state_builder.state)

        info: dict[str, Any] = {}

        self.logger.info(
            "new step number: %s", self.state_builder.state_counters["step"]
        )

        return (
            self.state_builder.state,
            reward,
            self.state_builder.terminated,
            False,
            info,
        )

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset environment state and counters for a new episode."""

        self.logger.info("env reset")

        self.state_builder.state_counters["step"] = 0

        self.logger.info("terminated %s", self.state_builder.terminated)
        self.logger.info("timed_out %s", self.state_builder.timed_out)
        self.logger.info(
            "new step number %s", self.state_builder.state_counters["step"]
        )

        if self.state_builder.terminated:
            self.state_builder.update_episode_counter()
            self.state_builder.custom_logic.reset_env_globals()

        if not self.state_builder.timed_out:
            self.state_builder.initialise_state()

        info: dict[str, Any] = {}

        return self.state_builder.state, info

    def render(self, mode: str = "human") -> None:
        """Render the environment."""

        self.logger.debug("render called with mode: %s", mode)

    def close(self) -> None:
        """Close any environment resources."""

        self.logger.debug("close called")

    @abstractmethod
    def _get_reward(self, action: int, state: dict[str, Any]) -> float:
        """Return the reward for the current step."""
