"""Simple text progress tracker (verbose=2)."""

import time

from .base import Progress
from .types import Phase, ProgressMetrics


class SimpleProgress(Progress):
    """Progress tracker that displays simple text output (verbose=2)."""

    __slots__ = (
        "_current_epoch",
        "_current_phase",
        "_epoch_start_time",
        "_losses",
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
        self._current_phase: Phase | None = None
        self._epoch_start_time = 0.0
        self._training_start_time = time.time()
        self._total_loss = 0.0
        self._total_samples = 0
        self._losses: dict[Phase, float] = {}

    def start_epoch(self, epoch: int) -> None:
        """Start a new epoch and record time."""
        self._current_epoch = epoch
        self._epoch_start_time = time.time()

    def start_phase(
        self,
        phase: Phase,
        total_batches: int = 0,  # noqa: ARG002
    ) -> None:
        """Start a new phase.

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
        """Update progress with metrics."""
        if metrics and "loss" in metrics and batch_size is not None:
            self._total_loss += metrics["loss"] * batch_size
            self._total_samples += batch_size

    def end_phase(self) -> float:
        """End the current phase and return average loss."""
        avg_loss = (
            self._total_loss / self._total_samples if self._total_samples > 0 else 0.0
        )
        match self._current_phase:
            case Phase.TRAIN:
                self._losses[Phase.TRAIN] = avg_loss
            case Phase.VALIDATION:
                self._losses[Phase.VALIDATION] = avg_loss
            case Phase.TEST:
                self._losses[Phase.TEST] = avg_loss
            case Phase.PREDICT:
                self._losses[Phase.PREDICT] = float("nan")
        return avg_loss

    def end_epoch(self) -> None:
        """End the current epoch and print summary."""
        epoch_time = time.time() - self._epoch_start_time
        epoch_num = self._current_epoch + 1

        match self._losses:
            case {Phase.TRAIN: train_loss, Phase.VALIDATION: val_loss}:
                print(
                    f"Epoch {epoch_num}/{self._total_epochs} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f} ({epoch_time:.2f}s)"
                )
            case {Phase.TRAIN: train_loss}:
                print(
                    f"Epoch {epoch_num}/{self._total_epochs} - "
                    f"Train Loss: {train_loss:.4f} ({epoch_time:.2f}s)"
                )
            case {Phase.TEST: test_loss}:
                print(f"Test Loss: {test_loss:.4f} ({epoch_time:.2f}s)")
            case {Phase.PREDICT: _}:
                print(f"Prediction completed ({epoch_time:.2f}s)")
        # Clear losses for next epoch
        self._losses.clear()

    def end_training(self) -> None:
        """End the training phase and print total time."""
        total_time = time.time() - self._training_start_time
        print(f"Training completed in {total_time:.2f}s")
