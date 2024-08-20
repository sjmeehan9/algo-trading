import importlib.util
import os
import sys

def wrapper_path(pipeline: dict) -> tuple:
    python_path = sys.path
    reward_wrapper_path = pipeline['pipeline']['model']['reward_wrapper_path']
    reward_wrapper_filename = pipeline['pipeline']['model']['reward_wrapper_filename']
    reward_function = 'calculate_reward'

    for path in python_path:
        temp_path = path + reward_wrapper_path + reward_wrapper_filename + '.py'
        if os.path.exists(temp_path):
            return temp_path, reward_wrapper_filename, reward_function
    
    return '', reward_wrapper_filename, reward_function


def wrapper_function(func: callable) -> callable:
    def wrapped_function(self, *args, **kwargs):
        if not hasattr(self, 'cached_reward'):
            reward_path, reward_wrapper_filename, reward_function = wrapper_path(self.pipeline)

            # Check if the custom file exists
            if os.path.exists(reward_path):
                # Dynamically import the custom module
                spec = importlib.util.spec_from_file_location(reward_wrapper_filename, reward_path)
                custom_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(custom_module)

                # Check if the custom implementation exists
                if hasattr(custom_module, reward_function):
                    self.logger.info(f'Using custom implementation from {reward_wrapper_filename}.py')
                    self.cached_reward = getattr(custom_module, reward_function)
                else:
                    self.logger.info(f'Custom reward function not found in {reward_wrapper_filename}.py, using default method')
                    self.cached_reward = func
            else:
                self.logger.info(f'{reward_wrapper_filename}.py not found, using default method')
                self.cached_reward = func

        # Call the cached method
        return self.cached_reward(self, *args, **kwargs)
    
    return wrapped_function