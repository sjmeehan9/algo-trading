from datetime import datetime
import json
import os
from pathlib import Path
import pytz
from .load_config import config_loader

# Function to check for an audit json file, and create one if it doesn't exist
def write_audit_json(audit_filepath: str, session_info: dict) -> None:
    # Check if the file exists
    if os.path.exists(audit_filepath):
        # File exists, read its content first
        with open(audit_filepath, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                # File exists but is empty or contains invalid JSON, start fresh
                data = {}
    else:
        # File does not exist, start with an empty dictionary
        data = {}
    
    # Update the data with new session_info
    data.setdefault('sessions', []).append(session_info)
    
    # Write the updated content back to the file
    with open(audit_filepath, 'w') as f:
        json.dump(data, f, indent=4)

    return None


# Function to find and return the pipeline config file
def pipeline_type_config(config_filename) -> dict:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = Path(current_dir).parents[0]

    config_file_path = os.path.join(parent_dir, 'config/', config_filename)
    config_file = config_loader(config_file_path, validate=False)

    return config_file


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
