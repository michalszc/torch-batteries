"""Utilities for calculating and managing metrics."""

from collections.abc import Callable
from typing import Any, Protocol, cast, runtime_checkable

import torch

from torch_batteries.utils.logging import get_logger

logger = get_logger("metrics")

type MetricCallable = Callable[[torch.Tensor, torch.Tensor], float | torch.Tensor]


@runtime_checkable
class StatefulMetric(Protocol):
    """Protocol for metrics computed from state accumulated over a full phase."""

    def reset(self) -> None:
        """Reset metric state before a phase."""

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metric state with one batch."""

    def compute(self) -> float | torch.Tensor:
        """Compute a scalar metric from accumulated state."""


class CollectedMetric(StatefulMetric):
    """Evaluate an ordinary metric callable once over a complete phase.

    This convenience adapter retains detached CPU predictions and targets, so its
    memory use grows with the dataset. Prefer an incremental ``StatefulMetric``
    implementation for large datasets.
    """

    __slots__ = ("_metric", "_predictions", "_targets")

    def __init__(self, metric: MetricCallable) -> None:
        self._metric = metric
        self._predictions: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []

    def reset(self) -> None:
        """Clear retained phase tensors."""
        self._predictions.clear()
        self._targets.clear()
        logger.debug("Collected metric state reset.")

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Retain one detached CPU batch."""
        self._predictions.append(predictions.detach().cpu())
        self._targets.append(targets.detach().cpu())
        logger.debug(
            "Collected metric retained batch: batches=%d, samples=%d",
            len(self._predictions),
            sum(_tensor_samples(item) for item in self._predictions),
        )

    def compute(self) -> float | torch.Tensor:
        """Concatenate retained tensors and evaluate the wrapped callable."""
        if not self._predictions:
            logger.error("Collected metric cannot compute without phase data.")
            msg = "CollectedMetric cannot compute without any updates."
            raise ValueError(msg)
        return self.compute_collected(
            torch.cat(self._predictions),
            torch.cat(self._targets),
        )

    def compute_collected(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> float | torch.Tensor:
        """Evaluate the wrapped callable over shared collected tensors."""
        return self._metric(predictions, targets)


type Metric = MetricCallable | StatefulMetric


def _tensor_samples(tensor: torch.Tensor) -> int:
    return tensor.shape[0] if tensor.ndim > 0 else 1


def _metric_float(name: str, value: Any) -> float:
    """Validate and normalize one computed metric value."""
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            logger.error("Metric '%s' returned a non-scalar tensor.", name)
            msg = f"Metric '{name}' must return a scalar value."
            raise ValueError(msg)
        return float(value.item())
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        logger.exception("Metric '%s' returned a non-numeric value.", name)
        msg = f"Metric '{name}' must return a numeric value."
        raise TypeError(msg) from error


class PhaseMetricManager:
    """Coordinate callable, incremental, and collected metrics for one phase."""

    __slots__ = (
        "_collected_predictions",
        "_collected_targets",
        "_failed",
        "_metrics",
    )

    def __init__(self, metrics: dict[str, Metric]) -> None:
        self._metrics = metrics
        self._collected_predictions: list[torch.Tensor] = []
        self._collected_targets: list[torch.Tensor] = []
        self._failed: set[str] = set()

    def reset(self) -> None:
        """Reset all phase-scoped metric state."""
        self._collected_predictions.clear()
        self._collected_targets.clear()
        self._failed.clear()
        for name, metric in self._metrics.items():
            if isinstance(metric, StatefulMetric):
                try:
                    metric.reset()
                    logger.debug("Stateful metric '%s' reset.", name)
                except Exception:  # noqa: BLE001
                    self._failed.add(name)
                    logger.warning(
                        "Failed to reset metric '%s'; skipping this phase.",
                        name,
                        exc_info=True,
                    )

    def update(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> dict[str, float]:
        """Update phase metrics and return per-batch callable values."""
        batch_values: dict[str, float] = {}
        metric_predictions = predictions.detach()
        metric_targets = targets.detach()
        collected_needed = any(
            isinstance(metric, CollectedMetric) and name not in self._failed
            for name, metric in self._metrics.items()
        )
        if collected_needed:
            self._collected_predictions.append(metric_predictions.cpu())
            self._collected_targets.append(metric_targets.cpu())
            logger.debug(
                "Shared metric collection updated: batches=%d, samples=%d",
                len(self._collected_predictions),
                sum(_tensor_samples(item) for item in self._collected_predictions),
            )

        for name, metric in self._metrics.items():
            if name in self._failed or isinstance(metric, CollectedMetric):
                continue
            try:
                if isinstance(metric, StatefulMetric):
                    metric.update(metric_predictions, metric_targets)
                    logger.debug("Stateful metric '%s' updated.", name)
                else:
                    batch_values[name] = _metric_float(
                        name, metric(metric_predictions, metric_targets)
                    )
            except Exception:  # noqa: BLE001
                self._failed.add(name)
                logger.warning(
                    "Failed to update metric '%s'; skipping this phase.",
                    name,
                    exc_info=True,
                )
        return batch_values

    def compute(self) -> dict[str, float]:
        """Compute all full-phase metric values."""
        results: dict[str, float] = {}
        shared_predictions: torch.Tensor | None = None
        shared_targets: torch.Tensor | None = None
        if self._collected_predictions:
            shared_predictions = torch.cat(self._collected_predictions)
            shared_targets = torch.cat(self._collected_targets)

        for name, metric in self._metrics.items():
            if name in self._failed or not isinstance(metric, StatefulMetric):
                continue
            try:
                if isinstance(metric, CollectedMetric):
                    if shared_predictions is None or shared_targets is None:
                        logger.error("Collected metric '%s' has no phase data.", name)
                        continue
                    value = metric.compute_collected(shared_predictions, shared_targets)
                else:
                    value = metric.compute()
                results[name] = _metric_float(name, value)
                logger.debug("Full-phase metric '%s' computed: %s", name, results[name])
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Failed to compute metric '%s'; skipping this phase.",
                    name,
                    exc_info=True,
                )
        return results

    def state_dict(self) -> dict[str, Any]:
        """Return optional states exposed by configured metric objects."""
        states: dict[str, Any] = {}
        for name, metric in self._metrics.items():
            state_method = getattr(metric, "state_dict", None)
            if callable(state_method):
                states[name] = state_method()
        logger.debug("Serialized %d metric states.", len(states))
        return states

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore optional configured metric states strictly by name."""
        expected = {
            name
            for name, metric in self._metrics.items()
            if callable(getattr(metric, "load_state_dict", None))
        }
        if set(state_dict) != expected:
            logger.error(
                "Metric checkpoint state mismatch: expected=%s, actual=%s",
                sorted(expected),
                sorted(state_dict),
            )
            msg = "Configured metric states do not match checkpoint state."
            raise ValueError(msg)
        for name in expected:
            metric = cast("Any", self._metrics[name])
            metric.load_state_dict(state_dict[name])
        logger.info("Restored %d metric states.", len(expected))


def calculate_metrics(
    metrics: dict[str, MetricCallable],
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
            calculated_metrics[metric_name] = _metric_float(metric_name, metric_value)
            logger.debug(
                "Calculated metric '%s': %s",
                metric_name,
                calculated_metrics[metric_name],
            )

        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Failed to calculate metric '%s': %s. Skipping this metric.",
                metric_name,
                e,
            )

    return calculated_metrics
