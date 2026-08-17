"""Internal metadata helpers for charged callables."""

from collections.abc import Callable
from typing import Any

from .core import Event


def get_charged_events(fn: Callable[..., Any]) -> tuple[Event, ...]:
    """Return all events attached to a callable in decorator application order."""
    events = getattr(fn, "_torch_batteries_events", None)
    if events is not None:
        return tuple(events)
    legacy_event = getattr(fn, "_torch_batteries_event", None)
    return () if legacy_event is None else (legacy_event,)
