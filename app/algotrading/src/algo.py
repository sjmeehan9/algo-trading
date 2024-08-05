import datetime
import logging
import os
from .initialise import init_task, init_pipeline, task_options
from .load_config import config_loader
from .log_setup import setup_logger

class AlgoTrading:
    CONFIG_FILENAME = 'config.yml'

    def __init__(self, main_dir: str) -> None:
        self.main_dir = main_dir
        
    def run(self) -> None:
        config_file_path = os.path.join(self.main_dir, self.CONFIG_FILENAME)

        # Load config
        config = config_loader(config_file_path)

        # Pipeline initialization
        pipeline = init_pipeline(config, self.main_dir)

        # Setup logger
        setup_logger(config['log_path'])
        logger = logging.getLogger(__name__)   
        logger.setLevel(logging.INFO)
        logger.info('Starting application, time is %s', datetime.datetime.now())

        # Init task
        init_task(config, task_options(), pipeline)

        logger.info('Algotrading session for this task has ended')

        return None