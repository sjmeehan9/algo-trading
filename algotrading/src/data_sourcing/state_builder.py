import logging

class StateBuilder:
    def __init__(self, config: dict, pipeline: dict):
        self.logger = logging.getLogger(__name__)

        # Assumes we're training an RL model
        
        # Build function to gather and read all the required data
        # Read to a dictionary then altogether into a dataframe
        # Store in a self variable and do not return anything
        # Add a data trim percentage as what start/end part of the data to drop
        # Keep model data columns in pipeline config
        
        # Build function to calculate episode length using StateBuilder data
        # This has already been calculated in check dataframe function in save_historical.py
        # Needs to be file - file_trim - number of rows used in the starting obs space

        # Build function to setup the state for step in the environment
        # Returns the window of data for the next step
        # Adds the custom reward variables to the state
        # Tracking of what place in the data we are is kept in StateBuilder

        # Placeholder for statebuilder function to process live data
