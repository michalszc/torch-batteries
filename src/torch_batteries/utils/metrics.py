"""Utilities for calculating and managing metrics."""

from collections.abc import Callable

import torch

from torch_batteries.utils.logging import get_logger

logger = get_logger("metrics")


def calculate_metrics(
    metrics: dict[str, Callable[[torch.Tensor, torch.Tensor], float | torch.Tensor]],
    pred: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """Calculate multiple metrics for given predictions and targets.

    This function takes a dictionary of metric functions and applies them to the
    predictions and targets. Each metric function should accept two tensors
    (predictions and targets) and return a scalar value (either as a tensor or float).

    The function handles both tensor and scalar returns from metric functions,
    automatically converting tensors to Python floats using `.item()`.

    If a metric function raises an exception during calculation, the error is logged
    as a warning and the metric is skipped (not included in the returned dictionary).

    Args:
        metrics: Dictionary mapping metric names to callable functions.
                Each function should have signature: fn(pred, target) -> float | Tensor
        pred: Model predictions as a tensor
        target: Ground truth target values as a tensor

    Returns:
        Dictionary mapping metric names to their calculated float values.
        Only successfully calculated metrics are included.

    Examples:
        ```python
         import torch.nn.functional as F

         def mae(pred, target):
             return F.l1_loss(pred, target)

         def rmse(pred, target):
             return torch.sqrt(F.mse_loss(pred, target))

         metrics_dict = {'mae': mae, 'rmse': rmse}
         pred = torch.tensor([[1.0], [2.0], [3.0]])
         target = torch.tensor([[1.1], [2.2], [2.9]])

         results = calculate_metrics(metrics_dict, pred, target)
         # returns: {'mae': 0.133..., 'rmse': 0.141...}
        ```

    Note:
        - Metric functions should not modify the input tensors
        - Both pred and target should have compatible shapes for the metric functions
        - Failed metric calculations are logged but don't raise exceptions
    """
    calculated_metrics = {}

    for metric_name, metric_fn in metrics.items():
        try:
            metric_value = metric_fn(pred, target)

            # Handle both tensor and scalar returns
            if isinstance(metric_value, torch.Tensor):
                calculated_metrics[metric_name] = metric_value.item()
            else:
                calculated_metrics[metric_name] = float(metric_value)

        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Failed to calculate metric '%s': %s. Skipping this metric.",
                metric_name,
                e,
            )

    return calculated_metrics
