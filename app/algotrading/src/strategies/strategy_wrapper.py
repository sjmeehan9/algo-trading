import importlib.util
import os
import sys

def wrapper_path(pipeline: dict) -> tuple:
    python_path = sys.path
    strategy_wrapper_path = pipeline['pipeline']['strategy']['strategy_wrapper_path']
    strategy_wrapper_filename = pipeline['pipeline']['strategy']['strategy_wrapper_filename']
    strategy_function = 'predict'

    for path in python_path:
        temp_path = path + strategy_wrapper_path + strategy_wrapper_filename + '.py'
        if os.path.exists(temp_path):
            return temp_path, strategy_wrapper_filename, strategy_function
    
    return '', strategy_wrapper_filename, strategy_function


def strategy_wrapper_function(func: callable) -> callable:
    def wrapped_function(self, *args, **kwargs):
        if not hasattr(self, 'cached_strategy'):
            strategy_path, strategy_wrapper_filename, strategy_function = wrapper_path(self.pipeline)

            # Check if the custom file exists
            if os.path.exists(strategy_path):
                # Dynamically import the custom module
                spec = importlib.util.spec_from_file_location(strategy_wrapper_filename, strategy_path)
                custom_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(custom_module)

                # Check if the custom implementation exists
                if hasattr(custom_module, strategy_function):
                    self.logger.info(f'Using custom implementation from {strategy_wrapper_filename}.py')
                    self.cached_strategy = getattr(custom_module, strategy_function)
                else:
                    self.logger.info(f'Custom strategy function not found in {strategy_wrapper_filename}.py, using default method')
                    self.cached_strategy = func
            else:
                self.logger.info(f'{strategy_wrapper_filename}.py not found, using default method')
                self.cached_strategy = func

        # Call the cached method
        return self.cached_strategy(self, *args, **kwargs)
    
    return wrapped_function