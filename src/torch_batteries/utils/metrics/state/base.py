"""Protocol for metrics that accumulate state across a phase."""

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class StatefulMetric(Protocol):
    """Protocol for metrics computed from state accumulated over a full phase.

    ``Battery`` calls ``reset`` before each phase, ``update`` with detached
    predictions and targets for each batch, then ``compute`` once. Implementations
    should return one numeric scalar.
    """

    def reset(self) -> None:
        """Reset metric state before a phase."""

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metric state with one batch.

        Args:
            predictions: Detached model predictions for the batch.
            targets: Detached targets for the batch.
        """

    def compute(self) -> float | torch.Tensor:
        """Compute a scalar metric from accumulated state."""
