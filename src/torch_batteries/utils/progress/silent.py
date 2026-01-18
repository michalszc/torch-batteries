"""Silent progress tracker (verbose=0)."""

from typing import cast

from .base import Progress
from .types import Phase, ProgressMetrics


class SilentProgress(Progress):
    """Progress tracker that produces no output (verbose=0).

    Tracks metrics internally but displays nothing to the console.
    Useful for production environments or when logging is handled externally.
    """

    __slots__ = (
        "_current_phase",
        "_total_metrics",
        "_total_samples",
    )

    def __init__(self, total_epochs: int = 1) -> None:  # noqa: ARG002
        """Initialize silent progress tracker.

        Args:
            total_epochs: Total number of epochs (unused, for interface compatibility).
        """
        self._total_metrics: dict[str, float] = {}
        self._total_samples = 0
        self._current_phase: Phase | None = None

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
        self._total_metrics = {}
        self._total_samples = 0

    def update(
        self, metrics: ProgressMetrics | None = None, batch_size: int | None = None
    ) -> None:
        """Update progress with batch metrics (silent, but tracked internally).

        Args:
            metrics: Dictionary of metrics for the current batch.
            batch_size: Number of samples in the batch for weighted averaging.
        """
        if metrics and batch_size is not None:
            for key, value in metrics.items():
                if key not in self._total_metrics:
                    self._total_metrics[key] = 0.0
                self._total_metrics[key] += cast("float", value) * batch_size
            self._total_samples += batch_size

    def end_phase(self) -> dict[str, float]:
        """End the current phase and return average metrics."""
        if self._total_samples > 0:
            return {
                key: total / self._total_samples
                for key, total in self._total_metrics.items()
            }
        return {}

    def end_epoch(self) -> None:
        """End the current epoch (silent)."""
        pass  # noqa: PIE790

    def end_training(self) -> None:
        """End the training phase (silent)."""
        pass  # noqa: PIE790
