"""Silent progress tracker (verbose=0)."""

from .base import Progress
from .types import Phase, ProgressMetrics


class SilentProgress(Progress):
    """Progress tracker that produces no output (verbose=0)."""

    __slots__ = ("_total_loss", "_total_samples")

    def __init__(self, total_epochs: int = 1) -> None:  # noqa: ARG002
        """Initialize silent progress tracker.

        Args:
            total_epochs: Total number of epochs (unused, for interface compatibility).
        """
        self._total_loss = 0.0
        self._total_samples = 0

    def start_epoch(self, epoch: int) -> None:
        """Start a new epoch (silent).

        Args:
            epoch: The epoch number (unused).
        """

    def start_phase(
        self,
        phase: Phase,  # noqa: ARG002
        total_batches: int = 0,  # noqa: ARG002
    ) -> None:
        """Start a new phase (silent).

        Args:
            phase: The training phase (unused).
            total_batches: Total number of batches (unused).
        """
        self._total_loss = 0.0
        self._total_samples = 0

    def update(
        self, metrics: ProgressMetrics | None = None, batch_size: int | None = None
    ) -> None:
        """Update progress after processing a batch."""
        if metrics and "loss" in metrics and batch_size is not None:
            self._total_loss += metrics["loss"] * batch_size
            self._total_samples += batch_size

    def end_phase(self) -> float:
        """End the current phase and return average loss."""
        return (
            self._total_loss / self._total_samples if self._total_samples > 0 else 0.0
        )

    def end_epoch(self, train_loss: float, val_loss: float | None = None) -> None:
        """End the current epoch (silent)."""

    def end_training(self) -> None:
        """End the training phase (silent)."""
