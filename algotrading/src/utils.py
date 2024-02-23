from datetime import datetime
import pytz

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