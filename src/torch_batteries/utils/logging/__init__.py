"""Centralized logging utilities for torch-batteries."""

from . import _functions
from ._functions import (
    disable_default_handler,
    enable_default_handler,
    enable_explicit_format,
    get_logger,
    reset_format,
    set_verbosity,
    set_verbosity_debug,
    set_verbosity_error,
    set_verbosity_info,
    set_verbosity_warning,
)

_manager = _functions._manager  # noqa: SLF001

__all__ = [
    "disable_default_handler",
    "enable_default_handler",
    "enable_explicit_format",
    "get_logger",
    "reset_format",
    "set_verbosity",
    "set_verbosity_debug",
    "set_verbosity_error",
    "set_verbosity_info",
    "set_verbosity_warning",
]
