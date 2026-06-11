"""Core events and decorators for torch-batteries."""

from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar

import torch
from torch import nn
from typing_extensions import ParamSpec, TypedDict

if TYPE_CHECKING:
    import torch_batteries
from torch_batteries.utils.logging import get_logger

P = ParamSpec("P")
R = TypeVar("R")

logger = get_logger("events")


class EventContext(TypedDict, total=False):
    """Context dictionary passed to event handlers.

    Different events populate different keys. All keys are optional so the same
    type can describe training, validation, testing, and prediction events.

    Common keys:

    - `battery`: The `Battery` instance managing the workflow.
    - `model`: The model/module being trained, validated, tested, or used for
      prediction.
    - `optimizer`: The optimizer when available.
    - `batch`: Current batch data, usually a tuple or list of tensors.
    - `batch_idx`: Current batch index within the active phase.
    - `epoch`: Current epoch number.

    Loss keys:

    - `train_loss`: Current training loss for training step events.
    - `val_loss`: Current validation loss for validation step events.
    - `test_loss`: Current test loss for test events.
    - `loss`: Deprecated compatibility alias for the phase-specific loss key.

    Metric keys:

    - `train_metrics`: Current training batch or epoch metrics.
    - `val_metrics`: Current validation batch or epoch metrics.
    - `test_metrics`: Current test batch or final metrics.

    History keys:

    - `history_train_loss`: Training loss history for completed epochs.
    - `history_val_loss`: Validation loss history for completed epochs.
    - `history_train_metrics`: Training metric history for completed epochs.
    - `history_val_metrics`: Validation metric history for completed epochs.

    Prediction keys:

    - `predictions`: Model predictions from a prediction step or prediction run.
    """

    battery: "torch_batteries.Battery"
    model: nn.Module
    optimizer: torch.optim.Optimizer | None
    batch: Any
    batch_idx: int
    epoch: int
    loss: float
    train_loss: float
    val_loss: float
    test_loss: float
    predictions: torch.Tensor | list[Any]
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    history_train_loss: list[float]
    history_val_loss: list[float]
    history_train_metrics: dict[str, list[float]]
    history_val_metrics: dict[str, list[float]]


class Event(Enum):
    """Events that can be used with the @charge decorator.

    Events are triggered at different points during training/testing/prediction.
    Each event receives an `EventContext` with different available fields.

    ## Training Events

    - `BEFORE_TRAIN`: Called before training starts.
        - **Context**: `optimizer`

    - `AFTER_TRAIN`: Called after training completes.
        - **Context**: `optimizer`, `epoch`, `train_metrics`,
          `val_metrics` (if validation ran), `history_train_loss`,
          `history_val_loss`, `history_train_metrics`, `history_val_metrics`

    - `BEFORE_TRAIN_EPOCH`: Called before each training epoch.
        - **Context**: `optimizer`, `epoch`

    - `AFTER_TRAIN_EPOCH`: Called after each training epoch.
        - **Context**: `optimizer`, `epoch`, `train_metrics`,
          `history_train_loss`, `history_val_loss`, `history_train_metrics`,
          `history_val_metrics`

    - `BEFORE_TRAIN_STEP`: Called before each training batch.
        - **Context**: `optimizer`, `batch`, `batch_idx`, `epoch`

    - `TRAIN_STEP`: Called for each training batch (must return loss).
        - **Context**: `optimizer`, `batch`, `batch_idx`, `epoch`

    - `AFTER_TRAIN_STEP`: Called after each training batch.
        - **Context**: `optimizer`, `batch`, `batch_idx`, `epoch`, `loss`,
          `train_loss`, `train_metrics`

    ## Validation Events

    - `BEFORE_VALIDATION`: Called before validation starts.
        - **Context**: `optimizer`, `epoch`, `train_metrics`,
          `history_train_loss`, `history_val_loss`, `history_train_metrics`,
          `history_val_metrics`

    - `AFTER_VALIDATION`: Called after validation completes.
        - **Context**: `optimizer`, `epoch`, `train_metrics`, `val_metrics`,
          `history_train_loss`, `history_val_loss`, `history_train_metrics`,
          `history_val_metrics`

    - `BEFORE_VALIDATION_EPOCH`: Called before each validation epoch.
        - **Context**: `epoch`

    - `AFTER_VALIDATION_EPOCH`: Called after each validation epoch.
        - **Context**: `epoch`, `val_metrics`

    - `BEFORE_VALIDATION_STEP`: Called before each validation batch.
        - **Context**: `batch`, `batch_idx`, `epoch`

    - `VALIDATION_STEP`: Called for each validation batch (must return loss).
        - **Context**: `batch`, `batch_idx`, `epoch`

    - `AFTER_VALIDATION_STEP`: Called after each validation batch.
        - **Context**: `batch`, `batch_idx`, `epoch`, `loss`, `val_loss`,
          `val_metrics`

    ## Test Events

    - `BEFORE_TEST`: Called before testing starts.
        - **Context**: `optimizer`

    - `AFTER_TEST`: Called after testing completes.
        - **Context**: `optimizer`, `loss`, `test_loss`, `test_metrics`

    - `BEFORE_TEST_EPOCH`: Called before test epoch.
        - **Context**: `optimizer`, `epoch`

    - `AFTER_TEST_EPOCH`: Called after test epoch.
        - **Context**: `optimizer`, `epoch`, `loss`, `test_loss`, `test_metrics`

    - `BEFORE_TEST_STEP`: Called before each test batch.
        - **Context**: `optimizer`, `batch`, `batch_idx`, `epoch`

    - `TEST_STEP`: Called for each test batch (must return loss).
        - **Context**: `optimizer`, `batch`, `batch_idx`, `epoch`

    - `AFTER_TEST_STEP`: Called after each test batch.
        - **Context**: `optimizer`, `batch`, `batch_idx`, `epoch`, `loss`,
          `test_loss`, `test_metrics`

    ## Prediction Events

    - `BEFORE_PREDICT`: Called before prediction starts.
        - **Context**: `optimizer`

    - `AFTER_PREDICT`: Called after prediction completes.
        - **Context**: `optimizer`, `predictions`

    - `BEFORE_PREDICT_EPOCH`: Called before prediction epoch.
        - **Context**: `optimizer`, `epoch`

    - `AFTER_PREDICT_EPOCH`: Called after prediction epoch.
        - **Context**: `optimizer`, `epoch`, `predictions`

    - `BEFORE_PREDICT_STEP`: Called before each prediction batch.
        - **Context**: `optimizer`, `batch`, `batch_idx`, `epoch`

    - `PREDICT_STEP`: Called for each prediction batch (must return predictions).
        - **Context**: `optimizer`, `batch`, `batch_idx`, `epoch`

    - `AFTER_PREDICT_STEP`: Called after each prediction batch.
        - **Context**: `optimizer`, `batch`, `batch_idx`, `epoch`, `predictions`
    """

    # Training lifecycle events
    BEFORE_TRAIN = "before_train"
    AFTER_TRAIN = "after_train"
    BEFORE_TRAIN_EPOCH = "before_train_epoch"
    AFTER_TRAIN_EPOCH = "after_train_epoch"
    BEFORE_TRAIN_STEP = "before_train_step"
    TRAIN_STEP = "train_step"
    AFTER_TRAIN_STEP = "after_train_step"

    # Validation lifecycle events
    BEFORE_VALIDATION = "before_validation"
    AFTER_VALIDATION = "after_validation"
    BEFORE_VALIDATION_EPOCH = "before_validation_epoch"
    AFTER_VALIDATION_EPOCH = "after_validation_epoch"
    BEFORE_VALIDATION_STEP = "before_validation_step"
    VALIDATION_STEP = "validation_step"
    AFTER_VALIDATION_STEP = "after_validation_step"

    # Test lifecycle events
    BEFORE_TEST = "before_test"
    AFTER_TEST = "after_test"
    BEFORE_TEST_EPOCH = "before_test_epoch"
    AFTER_TEST_EPOCH = "after_test_epoch"
    BEFORE_TEST_STEP = "before_test_step"
    TEST_STEP = "test_step"
    AFTER_TEST_STEP = "after_test_step"

    # Prediction lifecycle events
    BEFORE_PREDICT = "before_predict"
    AFTER_PREDICT = "after_predict"
    BEFORE_PREDICT_EPOCH = "before_predict_epoch"
    AFTER_PREDICT_EPOCH = "after_predict_epoch"
    BEFORE_PREDICT_STEP = "before_predict_step"
    PREDICT_STEP = "predict_step"
    AFTER_PREDICT_STEP = "after_predict_step"


def charge(event: Event) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to mark methods for specific training events.

    All event handlers should accept a single `EventContext` parameter containing
    relevant context for the event. Different events populate different fields.

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
        fn._torch_batteries_event = event  # type: ignore[attr-defined] # noqa: SLF001
        logger.info("Method '%s' charged with event '%s'", fn.__name__, event.value)
        return fn

    return decorator
