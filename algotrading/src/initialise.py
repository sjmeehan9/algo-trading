import os
from pathlib import Path
from threading import Timer
from .data_sourcing.save_historical import PastData
from .data_sourcing.live_streaming import LiveData
from .data_sourcing.state_builder import StateBuilder
from .load_config import pipeline_loader

# Init task functions
def init_task(config: dict, task_options: list, pipeline: dict) -> None:
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
        app.connect(config['ip_address'], config['port'], 0)
        Timer(app.timer, app.stop).start()
        app.run()
    elif task == 'task2':
        app = LiveData(pipeline)
        app.connect(config['ip_address'], config['port'], 0)
        Timer(app.timer, app.stop).start()
        app.run()
    elif task == 'task3':
        app = StateBuilder()
    elif task == 'task4':
        pass
    else:
        raise ValueError(f'{task} is not a valid task')

    return None


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