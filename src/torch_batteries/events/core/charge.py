"""Charged-event decorator."""

from collections.abc import Callable
from typing import TypeVar

from typing_extensions import ParamSpec

from torch_batteries.utils.logging import get_logger

from .event import Event

P = ParamSpec("P")
R = TypeVar("R")
logger = get_logger("events.core")


def charge(event: Event) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to mark methods for specific training events.

    Handlers accept the context associated with the selected event. Model and callback
    events receive `EventContext`; DataPack events receive `DataContext`.

    Args:
        event: The event type from the Event enum

    Returns:
        Decorated function with event metadata

    Examples:
        ```python
        @charge(Event.TRAIN_STEP)
        def training_step(self, context: EventContext):
            batch = context["batch"]
            x, y = batch
            pred = self(x)
            loss = F.mse_loss(pred, y)
            return loss

        @charge(Event.BEFORE_TRAIN_EPOCH)
        def on_epoch_start(self, context: EventContext):
            print(f"Starting epoch {context['epoch']}")

        @charge(Event.AFTER_TRAIN_STEP)
        def on_train_step_end(self, context: EventContext):
            # Log metrics, update learning rate, etc.
            if context.get("loss"):
                print(f"Batch {context['batch_idx']}: loss={context['loss']}")
        ```
    """

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        charged_events = getattr(fn, "_torch_batteries_events", ())
        fn._torch_batteries_events = (*charged_events, event)  # type: ignore[attr-defined] # noqa: SLF001
        fn._torch_batteries_event = event  # type: ignore[attr-defined] # noqa: SLF001
        logger.debug("Method '%s' charged with event '%s'", fn.__name__, event.value)
        return fn

    return decorator
