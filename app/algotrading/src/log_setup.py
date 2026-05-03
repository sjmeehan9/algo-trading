import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from algotrading.src.monitoring import StructuredLogFormatter


class JsonLogFormatter(StructuredLogFormatter):
    """JSON formatter for structured logging output."""


def setup_logger(
    path: str,
    print_logs: bool = False,
    log_level: int | str = logging.INFO,
    use_json: bool = True,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """Configure and return the application logger.

    Args:
        path: Directory path where log files should be stored.
        print_logs: If True, add a console stream handler.
        log_level: Logging level for handlers and root logger.
        use_json: If True, output logs as JSON-formatted records.
        max_bytes: Maximum log file size before rotation.
        backup_count: Number of rotated files to retain.

    Returns:
        Configured application logger instance.
    """
    log_dir = Path(path)
    log_dir.mkdir(parents=True, exist_ok=True)

    level = (
        logging.getLevelName(log_level.upper())
        if isinstance(log_level, str)
        else log_level
    )
    if not isinstance(level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    recfmt = "(%(threadName)s) %(asctime)s.%(msecs)03d %(levelname)s %(name)s %(filename)s:%(lineno)d %(message)s"
    timefmt = "%y%m%d_%H:%M:%S"
    formatter: logging.Formatter = (
        JsonLogFormatter(datefmt=timefmt)
        if use_json
        else logging.Formatter(fmt=recfmt, datefmt=timefmt)
    )

    log_file = log_dir / time.strftime("pyibapi.%y%m%d_%H%M%S.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    if print_logs:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    return logging.getLogger(__name__)
