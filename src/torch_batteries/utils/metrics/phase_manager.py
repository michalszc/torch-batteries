"""Utilities for calculating and managing metrics."""

from typing import Any, Literal, cast

import torch

from torch_batteries.utils.logging import get_logger

from ._helpers import _metric_float
from .metric_spec import MetricSpec
from .metric_types import Metric
from .state import CollectedMetric, StatefulMetric
from .types import MetricDefinition, Selection, StructuredTensors

logger = get_logger("utils.metrics")


class PhaseMetricManager:
    """Coordinate callable, incremental, and collected metrics for one phase.

    Ordinary callables produce batch values that are sample-weighted by progress
    tracking. Stateful metrics own their exact aggregation. ``CollectedMetric``
    instances share detached CPU collections. Metric lifecycle failures either
    propagate immediately or skip the failed metric for the remainder of the phase.

    Args:
        metrics: Named callable, stateful, or collected metrics.
        metric_error_policy: ``"raise"`` to propagate metric exceptions or
            ``"warn"`` to log and skip a failed metric for the current phase.
    """

    __slots__ = (
        "_collected_predictions",
        "_collected_targets",
        "_failed",
        "_metric_error_policy",
        "_metrics",
    )

    def __init__(
        self,
        metrics: dict[str, MetricDefinition],
        *,
        metric_error_policy: Literal["raise", "warn"] = "raise",
    ) -> None:
        if metric_error_policy not in {"raise", "warn"}:
            msg = "metric_error_policy must be either 'raise' or 'warn'."
            raise ValueError(msg)
        self._metrics = metrics
        self._metric_error_policy = metric_error_policy
        self._collected_predictions: dict[Selection, list[torch.Tensor]] = {}
        self._collected_targets: dict[Selection, list[torch.Tensor]] = {}
        self._failed: set[str] = set()

    @staticmethod
    def _metric(definition: MetricDefinition) -> Metric:
        """Unwrap a metric specification to its callable or stateful metric."""
        return definition.metric if isinstance(definition, MetricSpec) else definition

    @staticmethod
    def _selection(definition: MetricDefinition) -> Selection:
        """Get the prediction and target selectors for a metric."""
        if isinstance(definition, MetricSpec):
            return definition.predictions, definition.targets
        return None, None

    @staticmethod
    def _select(
        value: StructuredTensors, key: str | None, label: str, metric: str
    ) -> torch.Tensor:
        """Select one tensor from structured step values.

        Args:
            value: A tensor or top-level tensor mapping.
            key: Selected mapping key, or ``None`` for a plain tensor.
            label: Input label used in errors.
            metric: Metric name used in errors.
        """
        if key is None:
            if isinstance(value, torch.Tensor):
                return value
            msg = f"Metric '{metric}' requires a {label} selector."
            raise ValueError(msg)
        if not isinstance(value, dict):
            msg = f"Metric '{metric}' selected {label}='{key}' from a non-mapping."
            raise TypeError(msg)
        if key not in value:
            msg = f"Metric '{metric}' selected missing {label} key '{key}'."
            raise ValueError(msg)
        selected = value[key]
        if not isinstance(cast("object", selected), torch.Tensor):
            msg = f"Metric '{metric}' {label} key '{key}' must hold a tensor."
            raise TypeError(msg)
        return selected

    def reset(self) -> None:
        """Reset all phase-scoped metric state."""
        self._collected_predictions.clear()
        self._collected_targets.clear()
        self._failed.clear()
        for name, definition in self._metrics.items():
            metric = self._metric(definition)
            if isinstance(metric, StatefulMetric):
                try:
                    metric.reset()
                    logger.debug("Stateful metric '%s' reset.", name)
                except Exception:
                    if self._metric_error_policy == "raise":
                        logger.exception("Failed to reset metric '%s'.", name)
                        raise
                    self._failed.add(name)
                    logger.warning(
                        "Failed to reset metric '%s'; skipping this phase.",
                        name,
                        exc_info=True,
                    )

    def update(
        self, predictions: StructuredTensors, targets: StructuredTensors
    ) -> dict[str, float]:
        """Update phase metrics and return per-batch callable values.

        Args:
            predictions: Model predictions for the batch.
            targets: Targets for the batch.
        """
        batch_values: dict[str, float] = {}
        collected_in_batch: set[Selection] = set()
        for name, definition in self._metrics.items():
            if name in self._failed:
                continue
            metric = self._metric(definition)
            selection = self._selection(definition)
            try:
                metric_predictions = self._select(
                    predictions, selection[0], "predictions", name
                ).detach()
                metric_targets = self._select(
                    targets, selection[1], "targets", name
                ).detach()
                if isinstance(metric, CollectedMetric):
                    if selection not in collected_in_batch:
                        self._collected_predictions.setdefault(selection, []).append(
                            metric_predictions.cpu()
                        )
                        self._collected_targets.setdefault(selection, []).append(
                            metric_targets.cpu()
                        )
                        collected_in_batch.add(selection)
                elif isinstance(metric, StatefulMetric):
                    metric.update(metric_predictions, metric_targets)
                    logger.debug("Stateful metric '%s' updated.", name)
                else:
                    batch_values[name] = _metric_float(
                        name, metric(metric_predictions, metric_targets)
                    )
            except Exception:
                if self._metric_error_policy == "raise":
                    logger.exception("Failed to update metric '%s'.", name)
                    raise
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
        collected: dict[Selection, tuple[torch.Tensor, torch.Tensor]] = {
            selection: (torch.cat(items), torch.cat(self._collected_targets[selection]))
            for selection, items in self._collected_predictions.items()
            if items
        }

        for name, definition in self._metrics.items():
            metric = self._metric(definition)
            if name in self._failed or not isinstance(metric, StatefulMetric):
                continue
            try:
                if isinstance(metric, CollectedMetric):
                    selection = self._selection(definition)
                    if selection not in collected:
                        logger.error("Collected metric '%s' has no phase data.", name)
                        continue
                    value = metric.compute_collected(*collected[selection])
                else:
                    value = metric.compute()
                results[name] = _metric_float(name, value)
                logger.debug("Full-phase metric '%s' computed: %s", name, results[name])
            except Exception:
                if self._metric_error_policy == "raise":
                    logger.exception("Failed to compute metric '%s'.", name)
                    raise
                self._failed.add(name)
                logger.warning(
                    "Failed to compute metric '%s'; skipping this phase.",
                    name,
                    exc_info=True,
                )
        return results

    def state_dict(self) -> dict[str, Any]:
        """Return optional states exposed by configured metric objects."""
        states: dict[str, Any] = {}
        for name, definition in self._metrics.items():
            state_method = getattr(self._metric(definition), "state_dict", None)
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
            for name, definition in self._metrics.items()
            if callable(getattr(self._metric(definition), "load_state_dict", None))
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
            metric = cast("Any", self._metric(self._metrics[name]))
            metric.load_state_dict(state_dict[name])
        logger.info("Restored %d metric states.", len(expected))
