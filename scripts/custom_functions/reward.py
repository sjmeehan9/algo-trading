def calculate_reward(self, state: dict) -> float:
    reward = 2.0

    self.logger.info(f'step reward: {reward}')

    return reward