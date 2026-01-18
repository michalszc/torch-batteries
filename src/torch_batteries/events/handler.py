"""Event handler for managing decorated methods."""

from collections.abc import Callable
from typing import Any, ClassVar

from torch import nn

from torch_batteries.utils.logging import get_logger

from .core import Event

logger = get_logger("events.handler")


class EventHandler:
    """Handles discovery and execution of methods decorated with `@charge`.

    This class discovers methods on a model that are decorated with `@charge`
    and provides methods to call them based on events.

    Args:
        model: PyTorch model containing decorated methods
        callbacks: Optional list of callback objects with decorated methods

    Examples:
        ```python
        handler = EventHandler(model)
        loss = handler.call(Event.TRAIN_STEP, context)
        ```
    """

    MODEL_SPECIFIC_CALLBACKS: ClassVar[list[Event]] = [
        Event.TRAIN_STEP,
        Event.VALIDATION_STEP,
        Event.TEST_STEP,
        Event.PREDICT_STEP,
    ]

    def __init__(self, model: nn.Module, callbacks: list | None = None):
        self.model = model
        self._event_handlers: dict[Event, list[Callable] | Callable] = {}
        self._callbacks = callbacks
        self._discover_event_handlers()

    def _discover_event_handlers(self) -> None:
        """Discover methods decorated with @charge."""
        self._discover_model_event_handlers()
        self._discover_callback_event_handlers()

    def _discover_model_event_handlers(self) -> None:
        """Discover model-specific methods decorated with @charge."""
        discovered_count = 0

        for name in dir(self.model):
            method = getattr(self.model, name)
            if callable(method) and hasattr(method, "_torch_batteries_event"):
                event = method._torch_batteries_event  # noqa: SLF001
                if event in self.MODEL_SPECIFIC_CALLBACKS:
                    self._event_handlers[event] = method
                else:
                    if event not in self._event_handlers:
                        self._event_handlers[event] = []
                    if isinstance(self._event_handlers[event], list):
                        self._event_handlers[event].append(method)  # type: ignore[union-attr]
                discovered_count += 1
                logger.debug(
                    "Discovered handler '%s' for event '%s'", name, event.value
                )

        logger.info(
            "Discovered %d event handlers on model %s",
            discovered_count,
            type(self.model).__name__,
        )

    def _discover_callback_event_handlers(self) -> None:
        """Discover callback methods decorated with @charge."""

        if not self._callbacks:
            return

        discovered_count = 0

        for callback in self._callbacks:
            for name in dir(callback):
                method = getattr(callback, name)
                if callable(method) and hasattr(method, "_torch_batteries_event"):
                    event = method._torch_batteries_event  # noqa: SLF001
                    if event in self.MODEL_SPECIFIC_CALLBACKS:
                        logger.warning(
                            "Callback '%s' should not handle model-specific event '%s'",
                        )
                        continue
                    if event not in self._event_handlers:
                        self._event_handlers[event] = []
                    if isinstance(self._event_handlers[event], list):
                        self._event_handlers[event].append(method)  # type: ignore[union-attr]
                    discovered_count += 1
                    logger.debug(
                        "Discovered handler '%s' for event '%s' in callback '%s'",
                        name,
                        event.value,
                        type(callback).__name__,
                    )
        logger.info(
            "Discovered %d event handlers on %d callbacks",
            discovered_count,
            len(self._callbacks),
        )

    def get_handler(self, event: Event) -> list[Callable] | Callable | None:
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
            if isinstance(handler, list):
                logger.debug("Calling handlers for event '%s'", event.value)
                for h in handler:
                    h(*args, **kwargs)
                return None

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

    def get_handler_info(self) -> dict[Event, list[str] | str]:
        """Get information about all registered handlers.

        Returns:
            Dictionary mapping events to handler method names
        """
        result: dict[Event, list[str] | str] = {}
        for event, handler in self._event_handlers.items():
            if isinstance(handler, list):
                result[event] = [h.__name__ for h in handler]
            else:
                result[event] = handler.__name__
        return result
