"""Internal metadata helpers for charged callables."""

from collections.abc import Callable
from typing import Any

from .core import Event


def get_charged_events(fn: Callable[..., Any]) -> tuple[Event, ...]:
    """Return all events attached to a callable in decorator application order.

    Args:
        fn: Callable inspected for charge metadata.
    """
    return tuple(getattr(fn, "_torch_batteries_events", ()))
