"""Progress tracker for overall training management."""

import time


class ProgressTracker:
    """
    Progress tracker for training, validation, testing, and prediction phases.

    Supports different verbosity levels:
    - verbose=0: No output
    - verbose=1: Progress bars for each phase within epochs
    - verbose=2: Simple epoch-by-epoch loss printing

    Args:
        verbose: Verbosity level (0, 1, or 2)
        total_epochs: Total number of epochs (for training)
    """

    __slots__ = (
        "_current_epoch",
        "_epoch_start_time",
        "_start_time",
        "_total_epochs",
        "_train_loss",
        "_val_loss",
        "_verbose",
    )

    def __init__(
        self,
        verbose: int = 1,
        total_epochs: int = 1,
    ):
        self._verbose = verbose
        self._total_epochs = total_epochs
        self._current_epoch = 0
        self._start_time = time.time()
        self._epoch_start_time = time.time()
        self._train_loss = 0.0
        self._val_loss: float | None = None

    @property
    def verbose(self) -> int:
        """Get verbosity level."""
        return self._verbose

    @property
    def total_epochs(self) -> int:
        """Get total number of epochs."""
        return self._total_epochs

    @property
    def current_epoch(self) -> int:
        """Get current epoch."""
        return self._current_epoch

    @property
    def train_loss(self) -> float:
        """Get training loss."""
        return self._train_loss

    @train_loss.setter
    def train_loss(self, value: float) -> None:
        """Set training loss."""
        self._train_loss = value

    @property
    def val_loss(self) -> float | None:
        """Get validation loss."""
        return self._val_loss

    @val_loss.setter
    def val_loss(self, value: float | None) -> None:
        """Set validation loss."""
        self._val_loss = value

    def start_epoch(self, epoch: int) -> None:
        """Start a new epoch."""
        self._current_epoch = epoch
        self._epoch_start_time = time.time()

        if self._verbose == 1:
            print(f"Epoch {epoch + 1}/{self._total_epochs}")

    def end_epoch(self) -> None:
        """End the current epoch."""
        if self._verbose == 2:
            epoch_time = time.time() - self._epoch_start_time
            if self._val_loss is not None:
                print(
                    f"Epoch {self._current_epoch + 1}/{self._total_epochs} - "
                    f"Train Loss: {self._train_loss:.4f}, "
                    f"Val Loss: {self._val_loss:.4f} ({epoch_time:.2f}s)"
                )
            else:
                print(
                    f"Epoch {self._current_epoch + 1}/{self._total_epochs} - "
                    f"Train Loss: {self._train_loss:.4f} ({epoch_time:.2f}s)"
                )

    def end_phase(self) -> None:
        """End the current phase and clean up."""
        total_time = time.time() - self._start_time

        if self._verbose == 2:
            print(f"Training completed in {total_time:.2f}s")
