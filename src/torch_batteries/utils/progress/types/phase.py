"""Progress phase enumeration."""

from enum import Enum


class Phase(Enum):
    """Enumeration of training and evaluation phases."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    PREDICT = "predict"
