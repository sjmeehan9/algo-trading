import logging
import pandas as pd
from typing import Union
from .state_builder import StateBuilder

class StreamQueue:
    DATE_COLUMN = 'date'

    def __init__(self, config: dict, pipeline: dict):
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.pipeline = pipeline

        self.buffer_size = self.pipeline['pipeline']['model_data_config']['queue_buffer']
        self.pipeline_type = self.pipeline['pipeline']['model']['pipeline_type']

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
            self.route_function(self.queue.reset_index(drop=True))

        return None


    def __repr__(self) -> str:
        return f'StreamQueue({list(self.queue)})'
    

    def route(self) -> object:
        if self.pipeline_type == 'rl':
            self.state_builder = StateBuilder(self.config, self.pipeline)
            route = self.state_builder.live_data
        else:
            self.logger.error('Pipeline type not supported')
            route = None
        
        return route