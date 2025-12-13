"""Progress class with integrated factory for creating progress trackers."""

from .base import AbstractProgress
from .progress_bar import ProgressBarProgress
from .silent import SilentProgress
from .simple import SimpleProgress


class Progress:
    """
    Facade class for creating progress tracker instances.

    This class hides the factory pattern and provides a simple interface
    for creating the appropriate progress tracker based on verbosity level.

    Verbosity levels:
    - verbose=0: No output (silent) - SilentProgress
    - verbose=1: Progress bars for batches - ProgressBarProgress
    - verbose=2: Simple epoch-level text output - SimpleProgress

    Args:
        verbose: Verbosity level (0, 1, or 2)
        total_epochs: Total number of epochs (for training)

    Returns:
        The appropriate progress tracker instance.

    Raises:
        ValueError: If verbose level is not 0, 1, or 2.

    Example:
        ```python
        progress = Progress(verbose=1, total_epochs=10)

        for epoch in range(10):
            progress.start_epoch(epoch)
            progress.start_phase(Phase.TRAIN, total_batches=100)
            for batch in train_loader:
                progress.update({"loss": loss.item()}, batch_size)
            train_loss = progress.end_phase()
            progress.end_epoch(train_loss)
        progress.end_training()
        ```
    """

    def __new__(cls, verbose: int = 1, total_epochs: int = 1) -> AbstractProgress:
        """Create and return the appropriate progress tracker.

        Args:
            verbose: Verbosity level (0, 1, or 2).
            total_epochs: Total number of epochs.

        Returns:
            The appropriate progress tracker instance.

        Raises:
            ValueError: If verbose level is not 0, 1, or 2.
        """
        return cls.create(verbose=verbose, total_epochs=total_epochs)

    @staticmethod
    def create(verbose: int, total_epochs: int = 1) -> AbstractProgress:
        """Create a progress tracker based on verbosity level.

        Args:
            verbose: Verbosity level (0, 1, or 2).
            total_epochs: Total number of epochs.

        Returns:
            Progress tracker instance.

        Raises:
            ValueError: If verbose level is not 0, 1, or 2.
        """
        if verbose == 0:
            return SilentProgress(total_epochs)
        if verbose == 1:
            return ProgressBarProgress(total_epochs)
        if verbose == 2:
            return SimpleProgress(total_epochs)

        msg = f"Invalid verbose level: {verbose}. Must be 0, 1, or 2."
        raise ValueError(msg)
