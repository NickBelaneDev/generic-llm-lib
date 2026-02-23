"""Logging utilities for the generic LLM library."""

import logging
import sys

_LOGGER_NAME = "generic_llm_lib"


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance for the library.

    Args:
        name: Optional sub-logger name. If None, returns the root library logger.

    Returns:
        The requested logger.
    """
    if name:
        return logging.getLogger(f"{_LOGGER_NAME}.{name}")
    return logging.getLogger(_LOGGER_NAME)


def setup_logging(
    level: int = logging.INFO, format_str: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> None:
    """Setup default logging configuration for the library.

    This adds a StreamHandler to the library's root logger.
    Should typically be called by the application using the library, not the library itself,
    unless running as a standalone script.

    Args:
        level: Logging level.
        format_str: Log format string.
    """
    logger = logging.getLogger(_LOGGER_NAME)

    # Avoid adding multiple handlers if called multiple times
    if logger.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(format_str)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)


# Set default NullHandler to avoid "No handler found" warnings
logging.getLogger(_LOGGER_NAME).addHandler(logging.NullHandler())
