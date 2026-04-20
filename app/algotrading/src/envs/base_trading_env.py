"""Shared base environment for trading and strategy evaluation."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

import numpy as np
from algotrading.src.envs.signal_integration import SignalIntegration
from algotrading.src.trading.tools import TradingTools
from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete


class BaseTradingEnv(Env, ABC):
    """Base Gymnasium environment with shared trading logic."""

    DEFAULT_SPACE_MIN = 0
    DEFAULT_SPACE_MAX = 1000000
    ACTION_SPACE_SIZE = 3
    DUMMY_REWARD = 0

    def __init__(
        self,
        state_builder: object,
        signal_integration: SignalIntegration | None = None,
    ) -> None:
        super().__init__()

        self.logger = logging.getLogger(__name__)

        self.state_builder = state_builder
        self._signal_integration = signal_integration

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

        market_space = Dict(space_dict)
        if self._signal_integration is None:
            return market_space

        signal_low, signal_high = (
            self._signal_integration.get_observation_space_bounds()
        )
        signal_space = Box(low=signal_low, high=signal_high, dtype=np.float32)
        return Dict({"market": market_space, "signals": signal_space})

    def _build_observation(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build observation payload and optional signal debug metadata."""

        if self._signal_integration is None:
            return self.state_builder.state, {}

        current_timestamp = self._get_current_timestamp()
        signal_features, signal_info = self._signal_integration.get_signal_features(
            current_timestamp
        )
        observation: dict[str, Any] = {
            "market": self.state_builder.state,
            "signals": signal_features,
        }
        return observation, signal_info

    def _get_current_timestamp(self) -> datetime:
        """Resolve current state timestamp and ensure timezone awareness."""

        state_df = getattr(self.state_builder, "state_df", None)
        if state_df is not None and "date" in state_df and not state_df.empty:
            raw_timestamp = state_df["date"].iloc[-1]
            return self._to_aware_datetime(raw_timestamp)

        return datetime.now(timezone.utc)

    def _to_aware_datetime(self, value: Any) -> datetime:
        """Convert date-like value into timezone-aware datetime."""

        if isinstance(value, datetime):
            parsed = value
        elif hasattr(value, "to_pydatetime"):
            parsed = value.to_pydatetime()
        else:
            parsed = datetime.now(timezone.utc)

        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed

    def step(
        self, action: int
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Execute a single environment step."""

        self.logger.info("step taken")

        action = self.tools.stop_take(action, self.state_builder.state)

        self.state_builder.state_step(action)

        reward = self._get_reward(action, self.state_builder.state)

        observation, signal_info = self._build_observation()

        info: dict[str, Any] = {}
        if signal_info:
            info["signals"] = signal_info

        self.logger.info(
            "new step number: %s", self.state_builder.state_counters["step"]
        )

        return (
            observation,
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

        observation, _ = self._build_observation()

        info: dict[str, Any] = {}

        return observation, info

    def render(self, mode: str = "human") -> None:
        """Render the environment."""

        self.logger.debug("render called with mode: %s", mode)

    def close(self) -> None:
        """Close any environment resources."""

        self.logger.debug("close called")

    @abstractmethod
    def _get_reward(self, action: int, state: dict[str, Any]) -> float:
        """Return the reward for the current step."""
