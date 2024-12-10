import logging
import pandas as pd
from typing import Union
from ..reward_functions.reward import reward_factory
from .state_builder import StateBuilder
from ..strategies.custom_logic import custom_logic_factory

class StreamQueue:
    DATE_COLUMN = 'date'

    def __init__(self, config: dict, pipeline: dict):
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.pipeline = pipeline

        self.buffer_size = self.pipeline['pipeline']['state_data_config']['past_events']
        self.pipeline_type = self.pipeline['pipeline']['pipeline_type']

        self.queue = pd.DataFrame()
        self.dates = set()

        self.route_function = self.route()


    def put(self, data: Union[dict, list]) -> None:
        if isinstance(data, dict):
            data = [data]

        new_data = []
        for item in data:
            date = item.get(self.DATE_COLUMN)
            if date in self.dates:
                continue
            new_data.append(item)
            self.dates.add(date)

        if new_data:
            new_df = pd.DataFrame(new_data)
            self.queue = pd.concat([self.queue, new_df])

        # Sort the DataFrame by the date column
        self.queue.sort_values(by=self.DATE_COLUMN, inplace=True)

        # Ensure the queue does not exceed the buffer size
        if len(self.queue) > self.buffer_size:
            excess = len(self.queue) - self.buffer_size
            to_remove = self.queue.head(excess)
            self.dates -= set(to_remove[self.DATE_COLUMN])
            self.queue = self.queue.tail(self.buffer_size)

        if len(self.queue) == self.buffer_size:
            self.queue = self.queue.reset_index(drop=True)
            self.route_function(self.queue)

        return None


    def __repr__(self) -> str:
        return f'StreamQueue({list(self.queue)})'
    

    def route(self) -> object:
        if self.pipeline_type == 'rl':
            reward_name = self.pipeline['pipeline']['model']['model_reward']
            custom_logic = reward_factory(reward_name, self.config, self.pipeline)
        elif self.pipeline_type == 'strategy':
            strategy_name = self.pipeline['pipeline']['strategy']['strategy_name']
            custom_logic = custom_logic_factory(strategy_name, self.config, self.pipeline)
        else:
            self.logger.error('Pipeline type not supported to receive live data')
            raise NotImplementedError('Pipeline type not supported to receive live data')

        self.state_builder = StateBuilder(self.config, self.pipeline, custom_logic)
        route = self.state_builder.live_data
        
        self.logger.info(f'Route function defined')
        return route