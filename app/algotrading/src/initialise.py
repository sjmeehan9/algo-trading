import os
from threading import Timer
from .data_sourcing.save_historical import PastData
from .models.train_ml import TrainML
from .load_config import pipeline_loader
from .trading.trading_data import TradingStream

# Print the task options
def task_options() -> list:
    task_options = ['task1', 'task2', 'task3', 'task4']
    return task_options


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

    # Create data folder
    if not os.path.exists(config['data_path']):
        os.makedirs(config['data_path'])

    client_id = pipeline['pipeline']['client_id']

    # Run the selected task
    if task == 'task1':
        app = PastData(config, pipeline)
        app.connect(config['ip_address'], config['port'], client_id['historical'])
        Timer(app.timer, app.stop).start()
        app.run()
    elif task == 'task2':
        app = TrainML(config, pipeline)
        app.start()
    elif task == 'task3':
        app = TradingStream(config, pipeline, config['stream_data'])
        app.start()
    elif task == 'task4':
        app = TrainML(config, pipeline, True)
        app.start()
    else:
        raise ValueError(f'{task} is not a valid task')

    return None


# Init pipeline settings
def init_pipeline(config: dict, current_dir: str) -> dict:
    # Load pipeline settings json
    pipeline_name = config['pipeline']
    pipeline_filename = f'{pipeline_name}.json'
    pipeline_file_path = os.path.join(current_dir, 'pipeline_settings/', pipeline_filename)
    pipeline = pipeline_loader(pipeline_file_path)

    return pipeline