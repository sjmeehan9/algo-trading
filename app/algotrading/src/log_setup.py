import logging
import os
import time

# Create and configure logger
def setup_logger(path, print_logs=False) -> None:
    '''Function setup as many loggers as you want'''
    if not os.path.exists(path):
        os.makedirs(path)

    time.strftime('pyibapi.%Y%m%d_%H%M%S.log')

    recfmt = '(%(threadName)s) %(asctime)s.%(msecs)03d %(levelname)s %(filename)s:%(lineno)d %(message)s'

    timefmt = '%y%m%d_%H:%M:%S'

    logging.basicConfig(filename=time.strftime(f'{path}/pyibapi.%y%m%d_%H%M%S.log'),
                        filemode='w',
                        level=logging.INFO,
                        format=recfmt, datefmt=timefmt)
    logger = logging.getLogger(__name__)

    if print_logs:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logger.addHandler(console)

    return