"""Data types for trainer module."""

from typing import Any, TypedDict


class TrainResult(TypedDict):
    """Result from training process.

    Contains training and validation loss history.
    """

    train_loss: list[float]
    val_loss: list[float]


class TestResult(TypedDict):
    """Result from testing process.

    Contains test results from the test step handlers.
    """

    test_loss: float


class PredictResult(TypedDict):
    """Result from prediction process.

    Contains predictions from the predict step handlers.
    """

    predictions: list[Any]
