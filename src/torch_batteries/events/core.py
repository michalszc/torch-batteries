"""Core events and decorators for torch-batteries."""

from collections.abc import Callable
from enum import Enum
from typing import TypeVar

from typing_extensions import ParamSpec

from torch_batteries.utils.logging import get_logger

P = ParamSpec("P")
R = TypeVar("R")

logger = get_logger("events")


class Event(Enum):
    """Events that can be used with the @charge decorator."""

    # Training events
    TRAIN_STEP = "train_step"

    # Validation events
    VALIDATION_STEP = "validation_step"

    # Test events
    TEST_STEP = "test_step"

    # Prediction events
    PREDICT_STEP = "predict_step"


def charge(event: Event) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator to mark methods for specific training events.

    Args:
        event: The event type from the Event enum

    Returns:
        Decorated function with event metadata

    Example:
        ```python
        @charge(Event.TRAIN_STEP)
        def my_training_logic(self, batch):
            x, y = batch
            pred = self(x)
            loss = F.mse_loss(pred, y)
            return loss
        ```
    """

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        fn._torch_batteries_event = event  # type: ignore[attr-defined] # noqa: SLF001
        logger.info("Method '%s' charged with event '%s'", fn.__name__, event.value)
        return fn

    return decorator
