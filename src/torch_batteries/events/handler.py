"""Event handler for managing decorated methods."""

from collections.abc import Callable
from typing import Any

from torch import nn

from torch_batteries.utils.logging import get_logger

from .core import Event

logger = get_logger("events.handler")


class EventHandler:
    """
    Handles discovery and execution of methods decorated with @charge.

    This class discovers methods on a model that are decorated with @charge
    and provides methods to call them based on events.

    Args:
        model: PyTorch model containing decorated methods

    Example:
        ```python
        handler = EventHandler(model)
        loss = handler.call(Event.TRAIN_STEP, batch)
        ```
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._event_handlers: dict[Event, Callable] = {}
        self._discover_event_handlers()

    def _discover_event_handlers(self) -> None:
        """Discover methods decorated with @charge."""
        discovered_count = 0

        for name in dir(self.model):
            method = getattr(self.model, name)
            if callable(method) and hasattr(method, "_torch_batteries_event"):
                event = method._torch_batteries_event  # noqa: SLF001
                self._event_handlers[event] = method
                discovered_count += 1
                logger.debug(
                    "Discovered handler '%s' for event '%s'", name, event.value
                )

        logger.info(
            "Discovered %d event handlers on model %s",
            discovered_count,
            type(self.model).__name__,
        )

    def get_handler(self, event: Event) -> Callable | None:
        """Get the handler for a specific event.

        Args:
            event: The event to get a handler for

        Returns:
            The handler method if found, None otherwise
        """
        return self._event_handlers.get(event)

    def has_handler(self, event: Event) -> bool:
        """Check if a handler exists for the given event.

        Args:
            event: The event to check for

        Returns:
            True if a handler exists, False otherwise
        """
        return event in self._event_handlers

    def call(self, event: Event, *args: Any, **kwargs: Any) -> Any:
        """Call a handler if it exists.

        Args:
            event: The event to trigger
            *args: Positional arguments to pass to the handler
            **kwargs: Keyword arguments to pass to the handler

        Returns:
            The result of the handler call, or None if no handler exists
        """
        handler = self.get_handler(event)
        if handler:
            logger.debug("Calling handler for event '%s'", event.value)
            return handler(*args, **kwargs)
        logger.debug("No handler found for event '%s'", event.value)
        return None

    def get_all_events(self) -> list[Event]:
        """Get all events that have registered handlers.

        Returns:
            List of events that have handlers
        """
        return list(self._event_handlers.keys())

    def get_handler_info(self) -> dict[Event, str]:
        """Get information about all registered handlers.

        Returns:
            Dictionary mapping events to handler method names
        """
        return {
            event: handler.__name__ for event, handler in self._event_handlers.items()
        }
