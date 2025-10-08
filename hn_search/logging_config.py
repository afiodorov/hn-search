"""Centralized logging configuration for HN Search."""

import logging
import sys
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable


def setup_logging(level: str = "INFO"):
    """
    Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module."""
    return logging.getLogger(name)


@contextmanager
def log_time(logger: logging.Logger, operation: str, level: str = "INFO"):
    """
    Context manager to log execution time of an operation.

    Usage:
        with log_time(logger, "vector search"):
            # code here
    """
    start = time.time()
    logger.log(getattr(logging, level.upper()), f"⏱️  {operation} - starting")
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.log(
            getattr(logging, level.upper()),
            f"⏱️  {operation} - completed in {elapsed:.2f}s",
        )


def log_time_decorator(operation: str = None, level: str = "INFO"):
    """
    Decorator to log execution time of a function.

    Usage:
        @log_time_decorator("process query")
        def my_function():
            pass
    """

    def decorator(func: Callable) -> Callable:
        op_name = operation or f"{func.__module__}.{func.__name__}"

        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger = logging.getLogger(func.__module__)
            start = time.time()
            logger.log(getattr(logging, level.upper()), f"⏱️  {op_name} - starting")
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.time() - start
                logger.log(
                    getattr(logging, level.upper()),
                    f"⏱️  {op_name} - completed in {elapsed:.2f}s",
                )

        return wrapper

    return decorator


# Initialize logging on import
import os

log_level = os.environ.get("LOG_LEVEL", "INFO")
setup_logging(log_level)
