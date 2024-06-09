import logging
from ..models.predict import Predict

class Trading:
    def __init__(self, config: dict, pipeline: dict):
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.pipeline = pipeline

        self.predict = Predict(self.config, self.pipeline)
        

    def trading_algorithm(self, state: dict) -> None:
        action, _ = self.predict.get_action(state)

        action_int = action.item()

        return action_int