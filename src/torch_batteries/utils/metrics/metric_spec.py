"""Explicit input selection for structured step metrics."""

from dataclasses import dataclass
from typing import cast

from .metric_types import Metric


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """Bind a metric to top-level prediction and target keys.

    Args:
        metric: Callable, stateful, or collected metric.
        predictions: Key in a structured ``StepOutput.predictions`` mapping.
        targets: Key in a structured ``StepOutput.targets`` mapping.
    """

    metric: Metric
    predictions: str | None = None
    targets: str | None = None

    def __post_init__(self) -> None:
        """Reject incomplete or blank selectors."""
        if (self.predictions is None) != (self.targets is None):
            msg = "MetricSpec requires both predictions and targets selectors."
            raise ValueError(msg)
        for selector in (self.predictions, self.targets):
            if selector is not None and not isinstance(cast("object", selector), str):
                msg = "MetricSpec selectors must be strings."
                raise TypeError(msg)
            if selector is not None and not selector.strip():
                msg = "MetricSpec selectors must be non-blank strings."
                raise ValueError(msg)
