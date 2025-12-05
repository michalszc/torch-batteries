"""Types for progress tracking utilities."""

from enum import Enum
from typing import TypedDict


class Phase(Enum):
    """Enumeration of training phases."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    PREDICT = "predict"


class ProgressMetrics(TypedDict, total=False):
    """Metrics for progress tracking."""

    loss: float
