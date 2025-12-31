"""Simple text progress tracker (verbose=2)."""

import time
from typing import cast

from torch_batteries.utils.formatting import format_metrics

from .base import Progress
from .types import Phase, ProgressMetrics


class SimpleProgress(Progress):
    """Progress tracker that displays simple text output (verbose=2)."""

    __slots__ = (
        "_current_epoch",
        "_current_phase",
        "_epoch_start_time",
        "_phase_metrics",
        "_total_epochs",
        "_total_metrics",
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
        self._total_metrics: dict[str, float] = {}
        self._total_samples = 0
        self._phase_metrics: dict[Phase, dict[str, float]] = {}

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
        self._total_metrics = {}
        self._total_samples = 0

    def update(
        self, metrics: ProgressMetrics | None = None, batch_size: int | None = None
    ) -> None:
        """Update progress with metrics."""
        if metrics and batch_size is not None:
            for key, value in metrics.items():
                if key not in self._total_metrics:
                    self._total_metrics[key] = 0.0
                self._total_metrics[key] += cast("float", value) * batch_size
            self._total_samples += batch_size

    def end_phase(self) -> dict[str, float]:
        """End the current phase and return average metrics."""
        if self._total_samples > 0:
            avg_metrics = {
                key: total / self._total_samples
                for key, total in self._total_metrics.items()
            }
        else:
            avg_metrics = {}

        if self._current_phase:
            self._phase_metrics[self._current_phase] = avg_metrics
        return avg_metrics

    def end_epoch(self) -> None:
        """End the current epoch and print summary."""
        epoch_time = time.time() - self._epoch_start_time
        epoch_num = self._current_epoch + 1

        match self._phase_metrics:
            case {Phase.TRAIN: train_metrics, Phase.VALIDATION: val_metrics}:
                train_str = format_metrics(train_metrics, "Train ")
                val_str = format_metrics(val_metrics, "Val ")
                print(
                    f"Epoch {epoch_num}/{self._total_epochs} - "
                    f"{train_str}, {val_str} ({epoch_time:.2f}s)"
                )
            case {Phase.TRAIN: train_metrics}:
                train_str = format_metrics(train_metrics, "Train ")
                print(
                    f"Epoch {epoch_num}/{self._total_epochs} - "
                    f"{train_str} ({epoch_time:.2f}s)"
                )
            case {Phase.TEST: test_metrics}:
                test_str = format_metrics(test_metrics, "Test ")
                print(f"{test_str} ({epoch_time:.2f}s)")
            case {Phase.PREDICT: _}:
                print(f"Prediction completed ({epoch_time:.2f}s)")
        # Clear metrics for next epoch
        self._phase_metrics.clear()

    def end_training(self) -> None:
        """End the training phase and print total time."""
        total_time = time.time() - self._training_start_time
        print(f"Training completed in {total_time:.2f}s")
