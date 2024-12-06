import numpy as np
from typing import Dict, Tuple

class TradingStrategy:
    def __init__(self, data: Dict[str, np.ndarray]) -> None:
        self.data: Dict[str, np.ndarray] = data
        self.close: np.ndarray = data['close']
        self.current_position: int = int(data['current_position'][-1])  # Get the last known position

    def predict(self) -> np.ndarray:
        # - If no position (current_position == 0):
        #     If the most recent close is greater than the previous close, open a buy trade (action=1).
        #     Otherwise, open a short sell trade (action=2).
        #
        # - If currently long (current_position == 1):
        #     If the most recent close has dropped below the previous close, close the long by returning action=2 (which will close the long position).
        #     Otherwise, do nothing (action=0).
        #
        # - If currently short (current_position == 2):
        #     If the most recent close is above the previous close, close the short by returning action=1 (which will close the short position).
        #     Otherwise, do nothing (action=0).

        # Ensure we have at least two data points to compare price changes
        if len(self.close) < 2:
            return np.array([0], dtype=int)

        last_close = self.close[-1]
        prev_close = self.close[-2]

        if self.current_position == 0:
            # No position open
            if last_close > prev_close:
                # Price is going up, let's open a buy trade
                action = 1
            else:
                # Price is going down or staying flat, let's open a short sell trade
                action = 2
        elif self.current_position == 1:
            # Long position open
            if last_close < prev_close:
                # Price went down since last interval, close the long by going short (action=2)
                action = 2
            else:
                # Price did not drop, do nothing
                action = 0
        elif self.current_position == 2:
            # Short position open
            if last_close > prev_close:
                # Price went up since last interval, close the short by going long (action=1)
                action = 1
            else:
                # Price did not rise, do nothing
                action = 0
        else:
            # Undefined position scenario, just do nothing
            action = 0

        return np.array([action], dtype=int)


def predict(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict]:
    strategy = TradingStrategy(data)

    action = strategy.predict()
    _states = {}

    self.logger.info(f'Action: {action}')

    return action, _states