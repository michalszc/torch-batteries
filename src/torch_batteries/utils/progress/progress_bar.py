"""Progress bar tracker (verbose=1)."""

from typing import Any

from tqdm import tqdm

from .base import Progress
from .types import Phase, ProgressMetrics


class BarProgress(Progress):
    """Progress tracker that displays progress bars (verbose=1)."""

    __slots__ = (
        "_current_epoch",
        "_current_phase",
        "_pbar",
        "_total_epochs",
        "_total_loss",
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
        self._total_loss = 0.0
        self._total_samples = 0

    def start_epoch(self, epoch: int) -> None:
        """Start a new epoch and store epoch number."""
        self._current_epoch = epoch

    def start_phase(self, phase: Phase, total_batches: int = 0) -> None:
        """Start a new phase with progress bar."""
        self._current_phase = phase
        self._total_loss = 0.0
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
        """Update progress bar with metrics."""
        if metrics and "loss" in metrics and batch_size is not None:
            self._total_loss += metrics["loss"] * batch_size
            self._total_samples += batch_size

        if self._pbar:
            if self._total_samples > 0:
                avg_loss = self._total_loss / self._total_samples
                self._pbar.set_postfix_str(f"Loss={avg_loss:.4f}")
            self._pbar.update(1)

    def end_phase(self) -> float:
        """End the current phase and close progress bar."""
        if self._pbar:
            self._pbar.close()
            self._pbar = None

        return (
            self._total_loss / self._total_samples if self._total_samples > 0 else 0.0
        )

    def end_epoch(self) -> None:
        """End the current epoch (no output for verbose=1)."""
        pass  # noqa: PIE790

    def end_training(self) -> None:
        """End the training phase (no output for verbose=1)."""
        pass  # noqa: PIE790
