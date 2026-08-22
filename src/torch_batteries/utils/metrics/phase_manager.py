"""Utilities for calculating and managing metrics."""

from typing import Any, cast

import torch

from torch_batteries.utils.logging import get_logger

from ._helpers import _metric_float, _tensor_samples
from .metric_types import Metric
from .state import CollectedMetric, StatefulMetric

logger = get_logger("utils.metrics")


class PhaseMetricManager:
    """Coordinate callable, incremental, and collected metrics for one phase.

    Ordinary callables produce batch values that are sample-weighted by progress
    tracking. Stateful metrics own their exact aggregation. ``CollectedMetric``
    instances share detached CPU collections. A metric that raises is skipped for
    the remainder of the current phase.

    Args:
        metrics: Named callable, stateful, or collected metrics.
    """

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
                except Exception:
                    self._failed.add(name)
                    logger.warning(
                        "Failed to reset metric '%s'; skipping this phase.",
                        name,
                        exc_info=True,
                    )

    def update(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> dict[str, float]:
        """Update phase metrics and return per-batch callable values.

        Args:
            predictions: Model predictions for the batch.
            targets: Targets for the batch.
        """
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
            except Exception:
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
            except Exception:
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
        """Restore optional configured metric states strictly by name.

        Args:
            state_dict: Serialized state keyed by configured metric name.
        """
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
