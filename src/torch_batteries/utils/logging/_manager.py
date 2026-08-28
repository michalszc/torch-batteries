"""Package logger manager."""

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
