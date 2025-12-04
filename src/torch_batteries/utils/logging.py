"""Logging utilities for torch-batteries package.

This module provides a centralized logging system with configurable verbosity levels
and formatting options. The default log level is set to WARNING to reduce noise.
"""

import logging
import os
import sys
import threading

from torch_batteries.const import _PACKAGE_NAME


class _LoggerManager:
    """Manager for package logger configuration."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._default_handler: logging.StreamHandler | None = None

    @property
    def default_handler(self) -> logging.StreamHandler | None:
        """Get the default handler."""
        return self._default_handler

    def _get_default_logging_level(self) -> int:
        """Get the default logging level for the package.

        Returns:
            Logging level as an integer. Defaults to WARNING if not set via
            environment variable.

        Raises:
            ValueError: If the environment variable TORCH_BATTERIES_LOG_LEVEL is
                set to an invalid value.
        """
        env_level = os.getenv("TORCH_BATTERIES_LOG_LEVEL")

        match env_level:
            case None:
                return logging.WARNING
            case "DEBUG":
                return logging.DEBUG
            case "INFO":
                return logging.INFO
            case "WARNING":
                return logging.WARNING
            case "ERROR":
                return logging.ERROR
            case _:
                msg = f"Invalid log level: {env_level!r}"
                raise ValueError(msg)

    def _create_default_handler(self) -> logging.StreamHandler:
        """Create the default handler for the package logger.

        Returns:
            StreamHandler configured with default formatting and WARNING level.
        """
        handler = logging.StreamHandler(sys.stderr)

        formatter = logging.Formatter(
            fmt=f"[{_PACKAGE_NAME}] %(levelname)s: %(message)s"
        )
        handler.setFormatter(formatter)

        return handler

    def get_root_logger(self) -> logging.Logger:
        """Get the root package logger.

        Returns:
            The root logger for the package.
        """
        return logging.getLogger(_PACKAGE_NAME)

    def setup_logger(self) -> None:
        """Setup and configure the package logger with default handler."""
        with self._lock:
            if self._default_handler:
                # Already set up
                return

            logger = self.get_root_logger()
            logger.setLevel(self._get_default_logging_level())
            self._default_handler = self._create_default_handler()
            logger.addHandler(self._default_handler)

    def reset_default_handler(self) -> None:
        """Reset the default handler to its initial configuration.

        This recreates the default handler with original settings,
        useful for resetting any formatting or level changes.
        """
        with self._lock:
            if not self._default_handler:
                # Not set up yet
                return

            logger = self.get_root_logger()
            logger.removeHandler(self._default_handler)
            logger.setLevel(logging.NOTSET)
            self._default_handler = None


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
