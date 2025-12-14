"""Silent progress tracker (verbose=0)."""

from .base import Progress
from .types import Phase, ProgressMetrics


class SilentProgress(Progress):
    """Progress tracker that produces no output (verbose=0)."""

    __slots__ = (
        "_current_phase",
        "_total_loss",
        "_total_samples",
        "_train_loss",
        "_val_loss",
    )

    def __init__(self, total_epochs: int = 1) -> None:  # noqa: ARG002
        """Initialize silent progress tracker.

        Args:
            total_epochs: Total number of epochs (unused, for interface compatibility).
        """
        self._total_loss = 0.0
        self._total_samples = 0
        self._current_phase: Phase | None = None
        self._train_loss = 0.0
        self._val_loss: float | None = None

    def start_epoch(self, epoch: int) -> None:
        """Start a new epoch (silent).

        Args:
            epoch: The epoch number (unused).
        """
        pass  # noqa: PIE790

    def start_phase(
        self,
        phase: Phase,
        total_batches: int = 0,  # noqa: ARG002
    ) -> None:
        """Start a new phase (silent).

        Args:
            phase: The training phase.
            total_batches: Total number of batches (unused).
        """
        self._current_phase = phase
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

    def end_epoch(self) -> None:
        """End the current epoch (silent)."""
        pass  # noqa: PIE790

    def end_training(self) -> None:
        """End the training phase (silent)."""
        pass  # noqa: PIE790
