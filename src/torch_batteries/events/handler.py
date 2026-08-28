"""Event handler for managing decorated methods."""

from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from typing import Any, ClassVar, cast

from torch import nn

from torch_batteries.utils.logging import get_logger

from ._handler_base import _ChargedHandlerBase
from .core import Event

logger = get_logger("events.handler")


class EventHandler(_ChargedHandlerBase):
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
    EXCLUSIVE_EVENTS: ClassVar[set[Event]] = {
        Event.CONFIGURE_TRAIN_STEP,
        Event.BACKWARD,
        Event.GRADIENT_CLIP,
        Event.OPTIMIZER_STEP,
    }
    DATA_SPECIFIC_EVENTS: ClassVar[set[Event]] = {
        Event.PREPARE_DATA,
        Event.SETUP_DATA,
        Event.CONFIGURE_DATALOADER,
        Event.TEARDOWN_DATA,
    }

    def __init__(self, model: nn.Module, callbacks: list | None = None):
        super().__init__(logger)
        # Retain the private alias used by older integrations while the shared base
        # owns the canonical storage.
        self._event_handlers: dict[Event, list[Callable] | Callable] = cast(
            "dict[Event, list[Callable] | Callable]",
            self._handlers,
        )
        self.model = model
        self._callbacks = callbacks
        self._discover_event_handlers()

    def _discover_event_handlers(self) -> None:
        """Discover methods decorated with @charge."""
        self._discover_model_event_handlers()
        self._discover_callback_event_handlers()
        self._validate_exclusive_handlers()

    def _discover_model_event_handlers(self) -> None:
        """Discover model-specific methods decorated with @charge."""
        discovered_count = 0

        for name, method, event in self._discover_charged_methods(
            self.model,
            owner_description="Model",
        ):
            if event in self.DATA_SPECIFIC_EVENTS:
                msg = (
                    f"Model method '{name}' cannot handle DataPack event "
                    f"'{event.value}'."
                )
                raise ValueError(msg)
            if event in self.MODEL_SPECIFIC_CALLBACKS:
                self._set_single_handler(
                    event,
                    method,
                    name,
                    conflict_message=(
                        f"Event '{event.value}' accepts exactly one model handler; "
                        "found: {existing}, {new}."
                    ),
                )
            else:
                self._append_handler(event, method, f"model.{name}")
            discovered_count += 1
            logger.debug("Discovered handler '%s' for event '%s'", name, event.value)

        logger.debug(
            "Discovered %d event handlers on model %s",
            discovered_count,
            type(self.model).__name__,
        )

    def _discover_callback_event_handlers(self) -> None:
        """Discover callback methods decorated with @charge."""

        if not self._callbacks:
            return

        discovered_count = 0

        for callback_idx, callback in enumerate(self._callbacks):
            for name, method, event in self._discover_charged_methods(
                callback,
                owner_description=f"Callback '{type(callback).__name__}'",
            ):
                if event in self.DATA_SPECIFIC_EVENTS:
                    msg = (
                        f"Callback '{type(callback).__name__}' cannot handle "
                        f"DataPack event '{event.value}'."
                    )
                    raise ValueError(msg)
                if event in self.MODEL_SPECIFIC_CALLBACKS:
                    logger.warning(
                        "Callback '%s' should not handle model-specific event '%s'",
                        type(callback).__name__,
                        event.value,
                    )
                    continue
                self._append_handler(
                    event,
                    method,
                    f"callback[{callback_idx}].{type(callback).__name__}.{name}",
                )
                discovered_count += 1
                logger.debug(
                    "Discovered handler '%s' for event '%s' in callback '%s'",
                    name,
                    event.value,
                    type(callback).__name__,
                )
        logger.debug(
            "Discovered %d event handlers on %d callbacks",
            discovered_count,
            len(self._callbacks),
        )

    def _validate_exclusive_handlers(self) -> None:
        """Reject multiple owners for provider and executor events."""
        self._reject_conflicting_handlers(
            self.EXCLUSIVE_EVENTS,
            conflict_message=(
                "Event '{event}' accepts exactly one handler; found: {labels}."
            ),
        )

    def get_handler(self, event: Event) -> list[Callable] | Callable | None:
        """Get the handler for a specific event.

        Args:
            event: The event to get a handler for

        Returns:
            The handler method if found, None otherwise
        """
        handler = self._event_handlers.get(event)
        if handler is None:
            return None
        if event in self.MODEL_SPECIFIC_CALLBACKS and isinstance(handler, list):
            return handler[0]
        return handler

    def has_handler(self, event: Event) -> bool:
        """Check if a handler exists for the given event.

        Args:
            event: The event to check for.

        Returns:
            True if a handler exists, otherwise False.
        """
        return self._has_handler(event)

    def call(self, event: Event, *args: Any, **kwargs: Any) -> Any:
        """Call a handler if it exists.

        Args:
            event: The event to trigger
            *args: Positional arguments to pass to the handler
            **kwargs: Keyword arguments to pass to the handler

        Returns:
            The result of the handler call, or None if no handler exists
        """
        handlers = self._handlers_for(event)
        if event in self.MODEL_SPECIFIC_CALLBACKS and handlers:
            logger.debug("Calling handler for event '%s'", event.value)
            return handlers[0](*args, **kwargs)
        self._call_handlers(event, *args, require_none=False, **kwargs)
        return None

    def provide(
        self,
        event: Event,
        *args: Any,
        default: Any,
        **kwargs: Any,
    ) -> Any:
        """Return an exclusive provider result or a caller-supplied default.

        Args:
            event: Exclusive provider event to dispatch.
            *args: Positional arguments passed to the provider.
            default: Value returned when no provider is registered.
            **kwargs: Keyword arguments passed to the provider.
        """
        handler = self.get_handler(event)
        if handler is None:
            logger.debug(
                "No provider found for event '%s'; using default.", event.value
            )
            return default
        if not isinstance(handler, list) or len(handler) != 1:
            logger.error("Invalid provider registration for event '%s'.", event.value)
            msg = f"Event '{event.value}' requires one provider handler."
            raise ValueError(msg)
        logger.debug("Calling provider for event '%s'.", event.value)
        return handler[0](*args, **kwargs)

    def execute(self, event: Event, *args: Any, **kwargs: Any) -> bool:
        """Run one exclusive executor and report whether it handled the event.

        Args:
            event: Exclusive executor event to dispatch.
            *args: Positional arguments passed to the executor.
            **kwargs: Keyword arguments passed to the executor.
        """
        handler = self.get_handler(event)
        if handler is None:
            logger.debug("No executor found for event '%s'.", event.value)
            return False
        if not isinstance(handler, list) or len(handler) != 1:
            logger.error("Invalid executor registration for event '%s'.", event.value)
            msg = f"Event '{event.value}' requires one executor handler."
            raise ValueError(msg)
        logger.debug("Calling executor for event '%s'.", event.value)
        result = handler[0](*args, **kwargs)
        if result is not None:
            logger.error(
                "Executor for event '%s' returned %s instead of None.",
                event.value,
                type(result).__name__,
            )
            msg = f"Event '{event.value}' executor must return None."
            raise TypeError(msg)
        return True

    @contextmanager
    def execution_context(
        self,
        event: Event,
        *args: Any,
        **kwargs: Any,
    ) -> Generator[None]:
        """Enter every ordered context manager returned for an event.

        Args:
            event: Context-provider event to dispatch.
            *args: Positional arguments passed to each provider.
            **kwargs: Keyword arguments passed to each provider.
        """
        handlers = self._handlers_for(event)
        with ExitStack() as stack:
            for item in handlers:
                manager = item(*args, **kwargs)
                if not hasattr(manager, "__enter__") or not hasattr(
                    manager, "__exit__"
                ):
                    logger.error(
                        "Context provider for event '%s' returned %s.",
                        event.value,
                        type(manager).__name__,
                    )
                    msg = (
                        f"Event '{event.value}' handlers must return context managers."
                    )
                    raise TypeError(msg)
                stack.enter_context(cast("AbstractContextManager[Any]", manager))
            logger.debug(
                "Entered %d execution contexts for event '%s'.",
                len(handlers),
                event.value,
            )
            yield

    def get_all_events(self) -> list[Event]:
        """Get all events that have registered handlers.

        Returns:
            List of events that have handlers
        """
        return list(self._handlers.keys())

    def get_handler_info(self) -> dict[Event, list[str] | str]:
        """Get information about all registered handlers.

        Returns:
            Dictionary mapping events to handler method names
        """
        result: dict[Event, list[str] | str] = {}
        for event, handlers in self._handlers.items():
            if event in self.MODEL_SPECIFIC_CALLBACKS:
                result[event] = handlers[0].__name__
            else:
                result[event] = [handler.__name__ for handler in handlers]
        return result
