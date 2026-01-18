"""Types for progress tracking utilities."""

from enum import Enum
from typing import NotRequired, TypedDict


class Phase(Enum):
    """Enumeration of training phases.

    Used to track the current phase of model training/evaluation.

    Attributes:
        TRAIN: Training phase where model weights are updated
        VALIDATION: Validation phase for hyperparameter tuning
        TEST: Final evaluation phase on test set
        PREDICT: Inference phase for generating predictions
    """

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    PREDICT = "predict"


class ProgressMetrics(TypedDict, total=False):
    """Metrics for progress tracking.

    A flexible dictionary for storing training metrics at the batch level.
    The 'loss' field is the most common metric, but any additional metrics
    can be added dynamically.

    Attributes:
        loss: The loss value (typically required but marked as NotRequired
              for flexibility in prediction scenarios)

    Note:
        All fields are optional (total=False) to allow flexibility across
        different training scenarios. Additional metric fields can be added
        at runtime (e.g., 'accuracy', 'mae', 'rmse').

    Examples:
        ```python
        metrics: ProgressMetrics = {
            'loss': 0.5,
            'accuracy': 0.85,
            'mae': 0.23
        }
        ```
    """

    loss: NotRequired[float]
