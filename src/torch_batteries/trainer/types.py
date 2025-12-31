"""Data types for trainer module."""

from typing import Any, TypedDict


class TrainResult(TypedDict, total=False):
    """Result from training process.

    Contains training and validation loss history, plus optional custom metrics.
    """

    train_loss: list[float]
    val_loss: list[float]
    train_metrics: dict[str, list[float]]
    val_metrics: dict[str, list[float]]


class TestResult(TypedDict, total=False):
    """Result from testing process.

    Contains test results from the test step handlers, plus optional custom metrics.
    """

    test_loss: float
    test_metrics: dict[str, float]


class PredictResult(TypedDict):
    """Result from prediction process.

    Contains predictions from the predict step handlers.
    """

    predictions: list[Any]
