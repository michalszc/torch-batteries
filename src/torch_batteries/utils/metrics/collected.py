"""Utilities for calculating and managing metrics."""

from collections.abc import Callable

import torch

from torch_batteries.utils.logging import get_logger

from ._helpers import _tensor_samples
from .stateful import StatefulMetric

logger = get_logger("utils.metrics")

type MetricCallable = Callable[[torch.Tensor, torch.Tensor], float | torch.Tensor]


class CollectedMetric(StatefulMetric):
    """Evaluate an ordinary metric callable once over a complete phase.

    This convenience adapter retains detached CPU predictions and targets, so its
    memory use grows with the dataset. Prefer an incremental ``StatefulMetric``
    implementation for large datasets.

    Args:
        metric: Callable evaluated once with concatenated phase tensors.
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
        """Retain one detached CPU batch.

        Args:
            predictions: Model predictions for the batch.
            targets: Targets for the batch.
        """
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
        """Evaluate the wrapped callable over shared collected tensors.

        Args:
            predictions: Concatenated phase predictions.
            targets: Concatenated phase targets.
        """
        return self._metric(predictions, targets)
