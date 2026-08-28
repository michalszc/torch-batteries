"""Shared charged-handler discovery and dispatch infrastructure."""

from collections.abc import Callable, Generator, Iterable
from logging import Logger
from typing import Any

from ._metadata import get_charged_events
from .core import Event


class _ChargedHandlerBase:
    """Store and dispatch ordered methods decorated with ``@charge``."""

    def __init__(self, handler_logger: Logger) -> None:
        self._handlers: dict[Event, list[Callable[..., Any]]] = {}
        self._handler_labels: dict[Event, list[str]] = {}
        self._handler_logger = handler_logger

    def _discover_charged_methods(
        self,
        owner: object,
        *,
        owner_description: str,
    ) -> Generator[tuple[str, Callable[..., Any], Event]]:
        """Yield each charged method and reject duplicate stacked events."""
        for name in dir(owner):
            method = getattr(owner, name)
            if not callable(method):
                continue
            events = get_charged_events(method)
            if len(events) != len(set(events)):
                msg = (
                    f"{owner_description} method '{name}' is charged repeatedly "
                    "for one event."
                )
                raise ValueError(msg)
            for event in events:
                yield name, method, event

    def _append_handler(
        self,
        event: Event,
        handler: Callable[..., Any],
        label: str,
    ) -> None:
        """Append one ordered charged handler."""
        existing: Any = self._handlers.setdefault(event, [])
        if not isinstance(existing, list):
            self._handler_logger.error(
                "Cannot append handler to event %s with invalid storage.", event.value
            )
            msg = f"Cannot register multiple handlers for event '{event.value}'."
            raise TypeError(msg)
        existing.append(handler)
        self._handler_labels.setdefault(event, []).append(label)

    def _set_single_handler(
        self,
        event: Event,
        handler: Callable[..., Any],
        label: str,
        *,
        conflict_message: str,
    ) -> None:
        """Register one handler for an event that has a single owner."""
        existing = self._handler_labels.get(event, [])
        if existing:
            msg = conflict_message.format(existing=existing[0], new=label)
            raise ValueError(msg)
        self._append_handler(event, handler, label)

    def _reject_conflicting_handlers(
        self,
        events: Iterable[Event],
        *,
        conflict_message: str,
    ) -> None:
        """Reject events registered by more than one charged method."""
        for event in events:
            labels = self._handler_labels.get(event, [])
            if len(labels) <= 1:
                continue
            self._handler_logger.error(
                "Conflicting handlers for exclusive event '%s': %s",
                event.value,
                labels,
            )
            msg = conflict_message.format(
                event=event.value,
                labels=", ".join(labels),
            )
            raise ValueError(msg)

    def _handlers_for(self, event: Event) -> list[Callable[..., Any]]:
        """Return ordered handlers for an event."""
        return self._handlers.get(event, [])

    def _has_handler(self, event: Event) -> bool:
        """Return whether at least one handler is registered for an event."""
        return bool(self._handlers_for(event))

    def _call_handlers(
        self,
        event: Event,
        *args: Any,
        require_none: bool,
        **kwargs: Any,
    ) -> None:
        """Call every ordered handler for a side-effect event."""
        handlers = self._handlers_for(event)
        if not handlers:
            self._handler_logger.debug("No handler found for event '%s'", event.value)
            return
        self._handler_logger.debug("Calling handlers for event '%s'", event.value)
        for handler in handlers:
            result = handler(*args, **kwargs)
            if require_none and result is not None:
                msg = f"Event '{event.value}' handlers must return None."
                raise TypeError(msg)

    def _provide(
        self,
        event: Event,
        *args: Any,
        default: Any,
        invalid_message: str,
        **kwargs: Any,
    ) -> Any:
        """Return an exclusive provider result or a supplied default."""
        handlers = self._handlers_for(event)
        if not handlers:
            self._handler_logger.debug(
                "No provider found for event '%s'; using default.", event.value
            )
            return default
        if len(handlers) != 1:
            self._handler_logger.error(
                "Invalid provider registration for event '%s'.", event.value
            )
            raise ValueError(invalid_message)
        self._handler_logger.debug("Calling provider for event '%s'.", event.value)
        return handlers[0](*args, **kwargs)
