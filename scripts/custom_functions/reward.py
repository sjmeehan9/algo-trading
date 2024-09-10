import math

def calculate_reward(self, state: dict) -> float:
        base_reward = 0

        trade_change = state['trade_change'][-1]
        position_change = state['trade_change'][-2]
        running_change = state['running_profit'][-1]

        if trade_change > 0:
            trade_profit_reward = trade_change * 5
        elif trade_change < 0:
            trade_profit_reward = trade_change
        else:
            trade_profit_reward = 0

        if position_change > 0:
            position_reward = position_change * 500
        elif position_change < 0:
            position_reward = 0
        else:
            position_reward = 0

        if running_change > 0:
            running_profit = math.ceil(running_change)
            running_profit_reward = running_profit ** 3
        else:
            running_profit_reward = 0

        action_reward_dict = {
            'hold_nothing': 0.5,
            'hold_long_position': 2.5 + trade_profit_reward,
            'hold_short_position': 2.5 + trade_profit_reward,
            'buy_long': 1,
            'sell_short': 1,
            'sell_position': 1 + position_reward,
            'buyback_short': 1 + position_reward,
            'false_buy': 0,
            'false_sell': 0
        }

        action_reward = action_reward_dict[self.action_type]

        reward = base_reward + action_reward + running_profit_reward

        self.logger.info(f'step reward: {reward}')
        
        return reward