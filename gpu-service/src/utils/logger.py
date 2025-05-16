import logging
import sys
from typing import Optional, Union


def get_logger(
    name: str,
    level: Union[int, str] = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Name of the logger
        level: Logging level (default: INFO)
        format_string: Custom format string for the logger

    Returns:
        Configured logging.Logger instance
    """
    if format_string is None:
        format_string = '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'

    logger = logging.getLogger(name)

    # Only configure the logger if it hasn't been configured yet
    if not logger.handlers:
        logger.setLevel(level)

        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        # Create formatter
        formatter = logging.Formatter(format_string)
        handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(handler)

        # Prevent propagation to the root logger
        logger.propagate = False

    return logger
