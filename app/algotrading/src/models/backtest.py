"""Backtesting orchestration for RL and strategy pipelines."""

import logging
import os

from algotrading.src.trainers import RLTrainer

from ..strategies.strategy import Strategy
from ..utils import pipeline_type_config
from .train_ml import TrainML


class BackTest:
    """Manages backtest execution by routing to the appropriate trainer or strategy."""

    CONFIG_FILENAME = "pipeline_types.yml"

    def __init__(
        self,
        config: dict,
        pipeline: dict,
        trainer: RLTrainer | None = None,
    ):
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.pipeline = pipeline
        self.trainer = trainer

        self.pipeline_type = self.pipeline["pipeline"]["pipeline_type"]

        self.pipeline_type_dict = pipeline_type_config(self.CONFIG_FILENAME)

        self.path_dict = self._path_setup()

    def _path_setup(self) -> dict:
        path_dict = {}

        data_path = self.config["data_path"]
        backtest_data_path = os.path.join(data_path, "backtest/")
        path_dict["backtest_data_path"] = backtest_data_path

        self.pipeline_name = self.pipeline["pipeline"]["filename"]
        pipeline_backtest_path = os.path.join(
            backtest_data_path, f"{self.pipeline_name}/"
        )
        path_dict["pipeline_backtest_path"] = pipeline_backtest_path

        if not os.path.exists(backtest_data_path):
            os.makedirs(backtest_data_path)

        if not os.path.exists(pipeline_backtest_path):
            os.makedirs(pipeline_backtest_path)

        return path_dict

    def client(self, pipeline_type: str) -> object:
        if pipeline_type in self.pipeline_type_dict["pipeline_type"]["ml"]:
            return TrainML(
                self.config,
                self.pipeline,
                self.path_dict,
                True,
                trainer=self.trainer,
            )
        elif pipeline_type in self.pipeline_type_dict["pipeline_type"]["strategies"]:
            return Strategy(self.config, self.pipeline, self.path_dict, True)
        else:
            self.logger.error("Pipeline type not recognised")
            raise NotImplementedError("Pipeline type not recognised")

    def start(self) -> None:
        self.logger.info("Backtest initiated")

        client = self.client(self.pipeline_type)

        client.start()

        self.logger.info("Backtest completed")

        return None
