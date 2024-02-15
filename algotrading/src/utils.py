
from datetime import datetime
import os
from pathlib import Path
import pytz
from .load_config import pipeline_loader
from .data_sourcing.save_historical import PastData
from .data_sourcing.live_streaming import LiveData
from .data_sourcing.state_builder import StateBuilder


# Print the task options
def print_task_options() -> list:
    task_options = ['task1', 'task2', 'task3', 'task4']
    
    print('''
    Task Options:
    task1: Save batch historical data to csv
    task2: Train new/existing model using live or historical data
    task3: Run a trading session using a live or paper account
    task4: Run a backtest using historical data
    ''')

    return task_options


# Init task functions
def init_task(config: dict, task_options: list, pipeline: dict) -> object:
    # Show task selection and run task
    if config['task_selection']:
        task = config['task_selection']
    else:
        task = input('Please select a task: ')

        if task not in task_options:
            raise ValueError(f'{task} is not a valid task')
        
    task_message = f'Now running {task}'
    print(task_message)

    # Create log and data folders
    if not os.path.exists(config['log_path']):
        os.makedirs(config['log_path'])

    if not os.path.exists(config['data_path']):
        os.makedirs(config['data_path'])

    # Run the selected task
    if task == 'task1':
        app = PastData(config, pipeline)
    elif task == 'task2':
        app = LiveData(pipeline)
    elif task == 'task3':
        app = StateBuilder()
    elif task == 'task4':
        pass
    else:
        raise ValueError(f'{task} is not a valid task')

    return app


# Init pipeline settings
def init_pipeline(config: dict) -> dict:
    # Load pipeline settings json
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = Path(current_dir).parents[0]

    pipeline_name = config['pipeline']
    pipeline_filename = f'{pipeline_name}.json'
    pipeline_file_path = os.path.join(parent_dir, 'pipeline_settings/', pipeline_filename)
    pipeline = pipeline_loader(pipeline_file_path)

    return pipeline


# Custom function to parse datetime string and convert to a timezone-aware datetime object
def parse_datetime_tz(s: str) -> datetime:
    # Split the string to separate the timezone
    datetime_str, tz_str = s.rsplit(' ', 1)
    # Parse the datetime part
    dt = datetime.strptime(datetime_str, '%Y%m%d %H:%M:%S')
    # Localize the datetime object to the specified timezone
    tz = pytz.timezone(tz_str)
    dt = tz.localize(dt)

    return dt