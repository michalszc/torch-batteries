"""Data types for trainer module."""

from typing import Any, TypedDict


class TrainResult(TypedDict, total=False):
    """Result from training process.

    Contains training and validation loss history, plus optional custom metrics.

    Attributes:
        train_loss: List of average training loss values for each epoch
        val_loss: List of average validation loss values for each epoch
        train_metrics: Dictionary mapping metric names to lists of values per epoch
        val_metrics: Dictionary mapping metric names to lists of values per epoch

    Examples:
        ```python
        result: TrainResult = {
            'train_loss': [0.5, 0.3, 0.2],
            'val_loss': [0.6, 0.4, 0.3],
            'train_metrics': {'accuracy': [0.7, 0.8, 0.85]},
            'val_metrics': {'accuracy': [0.68, 0.78, 0.83]}
        }
        ```
    """

    train_loss: list[float]
    val_loss: list[float]
    train_metrics: dict[str, list[float]]
    val_metrics: dict[str, list[float]]


class TestResult(TypedDict, total=False):
    """Result from testing process.

    Contains test results from the test step handlers, plus optional custom metrics.

    Attributes:
        test_loss: Average test loss across all test batches
        test_metrics: Dictionary mapping metric names to their average values

    Example:
        ```python
        result: TestResult = {
            'test_loss': 0.25,
            'test_metrics': {'accuracy': 0.88, 'mae': 0.15}
        }
        ```
    """

    test_loss: float
    test_metrics: dict[str, float]


class PredictResult(TypedDict):
    """Result from prediction process.

    Contains predictions generated from the predict step handlers.

    Attributes:
        predictions: List of predictions from each batch. The format depends
                    on what the predict_step method returns (tensors, lists, etc.)

    Examples:
        ```python
        result: PredictResult = {
            'predictions': [tensor1, tensor2, tensor3, ...]
        }
        ```
    """

    predictions: list[Any]
