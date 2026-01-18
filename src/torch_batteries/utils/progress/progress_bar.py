"""Progress bar tracker (verbose=1)."""

from typing import Any, cast

from tqdm import tqdm

from .base import Progress
from .types import Phase, ProgressMetrics


class BarProgress(Progress):
    """Progress tracker that displays progress bars using tqdm (verbose=1).

    Shows a progress bar for each phase with real-time metrics updates.
    Provides visual feedback during training without verbose text output.
    """

    __slots__ = (
        "_current_epoch",
        "_current_phase",
        "_pbar",
        "_total_epochs",
        "_total_metrics",
        "_total_samples",
    )

    def __init__(self, total_epochs: int = 1) -> None:
        """Initialize progress bar tracker.

        Args:
            total_epochs: Total number of epochs.
        """
        self._total_epochs = total_epochs
        self._current_epoch = 0
        self._current_phase: Phase | None = None
        self._pbar: Any | None = None
        self._total_metrics: dict[str, float] = {}
        self._total_samples = 0

    def start_epoch(self, epoch: int) -> None:
        """Start a new epoch and store epoch number."""
        self._current_epoch = epoch

    def start_phase(self, phase: Phase, total_batches: int = 0) -> None:
        """Start a new phase with a progress bar.

        Args:
            phase: The training phase (train, validation, test, predict).
            total_batches: Total number of batches to process in this phase.
        """
        self._current_phase = phase
        self._total_metrics = {}
        self._total_samples = 0
        self._pbar = None

        if total_batches > 0:
            phase_name = self._current_phase.value.capitalize()
            epoch_num = self._current_epoch + 1
            desc = f"Epoch {epoch_num}/{self._total_epochs} [{phase_name}]"
            self._pbar = tqdm(
                total=total_batches,
                desc=desc,
                leave=True,
            )

    def update(
        self, metrics: ProgressMetrics | None = None, batch_size: int | None = None
    ) -> None:
        """Update progress bar with current batch metrics.

        Args:
            metrics: Dictionary of metrics (e.g., loss, accuracy) for the current batch.
            batch_size: Number of samples in the batch for weighted averaging.
        """
        if metrics and batch_size is not None:
            for key, value in metrics.items():
                if key not in self._total_metrics:
                    self._total_metrics[key] = 0.0
                self._total_metrics[key] += cast("float", value) * batch_size
            self._total_samples += batch_size

        if self._pbar:
            if self._total_samples > 0:
                # Display all metrics in progress bar
                postfix_parts = []
                for key, total_value in self._total_metrics.items():
                    avg_value = total_value / self._total_samples
                    metric_name = key.capitalize()
                    postfix_parts.append(f"{metric_name}={avg_value:.4f}")
                self._pbar.set_postfix_str(", ".join(postfix_parts))
            self._pbar.update(1)

    def end_phase(self) -> dict[str, float]:
        """End the current phase, close progress bar, and return average metrics.

        Returns:
            Dictionary of average metrics calculated across all batches in the phase.
        """
        if self._pbar:
            self._pbar.close()
            self._pbar = None

        if self._total_samples > 0:
            return {
                key: total / self._total_samples
                for key, total in self._total_metrics.items()
            }
        return {}

    def end_epoch(self) -> None:
        """End the current epoch (no output for verbose=1)."""
        pass  # noqa: PIE790

    def end_training(self) -> None:
        """End the training phase (no output for verbose=1)."""
        pass  # noqa: PIE790
