"""Logging utilities for torch-batteries package.

This module provides a centralized logging system with configurable verbosity levels
and formatting options. The default log level is set to WARNING to reduce noise.
"""

import logging

from torch_batteries.const import _PACKAGE_NAME

from ._manager import _LoggerManager

_manager = _LoggerManager()


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance for the torch-batteries package.

    Args:
        name: Optional name for the logger. If None, returns the package logger.
              If provided, returns a child logger.

    Returns:
        Logger instance configured with the package's default settings.
    """
    _manager.setup_logger()
    base_logger = _manager.get_root_logger()

    if name is None:
        return base_logger

    return base_logger.getChild(name)


def set_verbosity(level: int) -> None:
    """Set the verbosity level for the package logger.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO, etc.)
    """
    assert _manager.default_handler is not None, "Default handler is not set up."

    _manager.get_root_logger().setLevel(level)


def set_verbosity_info() -> None:
    """Set verbosity to `INFO` level."""
    set_verbosity(logging.INFO)


def set_verbosity_warning() -> None:
    """Set verbosity to `WARNING` level."""
    set_verbosity(logging.WARNING)


def set_verbosity_debug() -> None:
    """Set verbosity to `DEBUG` level."""
    set_verbosity(logging.DEBUG)


def set_verbosity_error() -> None:
    """Set verbosity to `ERROR` level."""
    set_verbosity(logging.ERROR)


def disable_default_handler() -> None:
    """Disable the default handler for the package logger.

    This allows users to configure their own logging setup without
    interference from the package's default handler.
    """
    assert _manager.default_handler is not None, "Default handler is not set up."

    logger = _manager.get_root_logger()
    logger.removeHandler(_manager.default_handler)


def enable_default_handler() -> None:
    """Enable the default handler for the package logger.

    Re-adds the default handler if it was previously disabled.
    """
    assert _manager.default_handler is not None, "Default handler is not set up."

    logger = _manager.get_root_logger()
    logger.addHandler(_manager.default_handler)


def enable_explicit_format() -> None:
    """Enable explicit formatting with timestamps and module information.

    Changes the log format to include timestamps, log levels, filenames,
    and line numbers.
    """
    assert _manager.default_handler is not None, "Default handler is not set up."

    explicit_formatter = logging.Formatter(
        "[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s"
    )
    _manager.default_handler.setFormatter(explicit_formatter)


def reset_format() -> None:
    """Reset the log format to the default simple format."""
    assert _manager.default_handler is not None, "Default handler is not set up."

    explicit_formatter = logging.Formatter(
        f"[{_PACKAGE_NAME}] %(levelname)s: %(message)s"
    )
    _manager.default_handler.setFormatter(explicit_formatter)
