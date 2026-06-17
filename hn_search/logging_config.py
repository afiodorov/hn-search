"""Centralized logging configuration for HN Search."""

import logging
import os
import sys
import time
from contextlib import contextmanager


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


# Initialize logging on import
log_level = os.environ.get("LOG_LEVEL", "INFO")
setup_logging(log_level)
