import logging

class TradingTools:
    ACTIONS = {1: 2, 2: 1}

    def __init__(self, pipeline: dict):
        self.logger = logging.getLogger(__name__)

        self.pipeline = pipeline


    def stop_take(self, action: int, state: dict) -> int:
        if not self.pipeline['pipeline']['trading_config']['stop_take']['enabled']:
            return action

        config = self.pipeline['pipeline']['trading_config']['stop_take']
        takeover_mode = config['takeover_mode']
        stop_loss_limit = config['stop_loss_limit']
        take_profit_rolling = config['take_profit_rolling']
        take_profit_floor = config['take_profit_floor']
        profit_key = config['profit_key']
        position_key = config['position_key']

        if state[position_key][-1] == 0:
            self.in_flight = False
            return action

        if stop_loss_limit <= state[profit_key][-1] <= take_profit_rolling and not self.in_flight:
            if takeover_mode:
                self.logger.info('Takeover mode enabled, keeping position open')
                return 0
            return action

        if (state[profit_key][-1] < stop_loss_limit or state[profit_key][-1] > take_profit_rolling) and not self.in_flight:
            self.in_flight = True
            if state[profit_key][-1] < stop_loss_limit:
                self.set_marker = None
                self.logger.info('Stop loss triggered, closing position')
                return self.ACTIONS[state[position_key][-1]]
            self.set_marker = state[profit_key][-1]
            self.logger.info('Starting rolling take profit, until floor is reached or position is closed')
            return action

        if self.in_flight:
            if self.set_marker is None:
                self.logger.info('Stop loss triggered, attempting to close position again')
                return self.ACTIONS[state[position_key][-1]]
            if state[profit_key][-1] > self.set_marker:
                self.set_marker = state[profit_key][-1]
                self.logger.info('New take profit marker set: %s', self.set_marker)
                return action
            if self.set_marker - state[profit_key][-1] >= take_profit_floor:
                self.logger.info('Take profit floor reached, closing position')
                return self.ACTIONS[state[position_key][-1]]
            return action