"""Abstract base class for progress tracking."""

from abc import ABC, abstractmethod

from .types import Phase, ProgressMetrics


class Progress(ABC):
    """Abstract base class for progress tracking during training."""

    @abstractmethod
    def start_epoch(self, epoch: int) -> None:
        """Start a new epoch.

        Args:
            epoch: The epoch number (0-indexed).
        """

    @abstractmethod
    def start_phase(self, phase: Phase, total_batches: int = 0) -> None:
        """Start a new phase (train, validation, test, predict).

        Args:
            phase: The current phase of training (e.g., train, validation,
                   test, predict).
            total_batches: Total number of batches in the phase.

        Returns:
            None
        """

    @abstractmethod
    def update(
        self, metrics: ProgressMetrics | None = None, batch_size: int | None = None
    ) -> None:
        """Update progress after processing a batch.

        Args:
            metrics: Optional metrics dictionary containing 'loss' and other metrics.
            batch_size: Optional batch size for averaging metrics.

        Returns:
            None
        """

    @abstractmethod
    def end_phase(self) -> float:
        """End the current phase and return average loss.

        Returns:
            Average loss across all batches, or 0.0 if no samples processed.
        """

    @abstractmethod
    def end_epoch(self) -> None:
        """End the current epoch and display summary."""

    @abstractmethod
    def end_training(self) -> None:
        """End the training phase and display total time.

        Returns:
            None
        """
