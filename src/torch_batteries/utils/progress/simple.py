"""Simple text progress tracker (verbose=2)."""

import time

from .base import Progress
from .types import Phase, ProgressMetrics


class SimpleProgress(Progress):
    """Progress tracker that displays simple text output (verbose=2)."""

    __slots__ = (
        "_current_epoch",
        "_epoch_start_time",
        "_total_epochs",
        "_total_loss",
        "_total_samples",
        "_training_start_time",
    )

    def __init__(self, total_epochs: int = 1) -> None:
        """Initialize simple text progress tracker.

        Args:
            total_epochs: Total number of epochs.
        """
        self._total_epochs = total_epochs
        self._current_epoch = 0
        self._epoch_start_time = 0.0
        self._training_start_time = time.time()
        self._total_loss = 0.0
        self._total_samples = 0

    def start_epoch(self, epoch: int) -> None:
        """Start a new epoch and record time."""
        self._current_epoch = epoch
        self._epoch_start_time = time.time()

    def start_phase(
        self,
        phase: Phase,  # noqa: ARG002
        total_batches: int = 0,  # noqa: ARG002
    ) -> None:
        """Start a new phase.

        Args:
            phase: The training phase (unused).
            total_batches: Total number of batches (unused).
        """
        self._total_loss = 0.0
        self._total_samples = 0

    def update(
        self, metrics: ProgressMetrics | None = None, batch_size: int | None = None
    ) -> None:
        """Update progress with metrics."""
        if metrics and "loss" in metrics and batch_size is not None:
            self._total_loss += metrics["loss"] * batch_size
            self._total_samples += batch_size

    def end_phase(self) -> float:
        """End the current phase and return average loss."""
        return (
            self._total_loss / self._total_samples if self._total_samples > 0 else 0.0
        )

    def end_epoch(self, train_loss: float, val_loss: float | None = None) -> None:
        """End the current epoch and print summary."""
        epoch_time = time.time() - self._epoch_start_time
        epoch_num = self._current_epoch + 1

        if val_loss is not None:
            print(
                f"Epoch {epoch_num}/{self._total_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f} ({epoch_time:.2f}s)"
            )
        else:
            print(
                f"Epoch {epoch_num}/{self._total_epochs} - "
                f"Train Loss: {train_loss:.4f} ({epoch_time:.2f}s)"
            )

    def end_training(self) -> None:
        """End the training phase and print total time."""
        total_time = time.time() - self._training_start_time
        print(f"Training completed in {total_time:.2f}s")
