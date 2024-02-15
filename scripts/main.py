import datetime
import logging
import os
from pathlib import Path
from threading import Timer

import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# TODO Remove two lines above once algotrading is a package

from algotrading import (config_loader,
    PastData,
    LiveData,
    setup_logger,
    StateBuilder,
    print_task_options,
    pipeline_loader,
    init_task,
    init_pipeline
)

config_filename = 'local_config.yml'

def main(filename: str) -> str:

    # Set current working directory
    main_dir = os.path.dirname(os.path.abspath(__file__))

    # Define config file path
    config_file_path = os.path.join(main_dir, filename)

    # Load config
    config = config_loader(config_file_path)

    task_options = print_task_options()

    # Pipeline initialization
    pipeline = init_pipeline(config)

    # Setup logger
    setup_logger(config['log_path'])
    logger = logging.getLogger(__name__)    
    logger.info("Starting application, time is %s", datetime.datetime.now())

    # Init task
    app = init_task(config, task_options, pipeline)
    
    app.connect(config['ip_address'], config['port'], 0)

    Timer(app.timer, app.stop).start()

    app.run()

    return print('algotrading session for this task has ended')
    

if __name__ == "__main__":
    main(config_filename)