"""Factory for creating progress trackers."""

from .base import Progress
from .progress_bar import BarProgress
from .silent import SilentProgress
from .simple import SimpleProgress


class ProgressFactory:
    """Factory for creating progress tracker instances."""

    @staticmethod
    def create(verbose: int, total_epochs: int = 1) -> Progress:
        """Create a progress tracker based on verbosity level.

        Args:
            verbose: Verbosity level (0, 1, or 2).
            total_epochs: Total number of epochs.

        Returns:
            Progress tracker instance.

        Raises:
            ValueError: If verbose level is not 0, 1, or 2.
        """
        match verbose:
            case 0:
                return SilentProgress(total_epochs)
            case 1:
                return BarProgress(total_epochs)
            case 2:
                return SimpleProgress(total_epochs)
            case _:
                msg = f"Invalid verbose level: {verbose}. Must be 0, 1, or 2."
                raise ValueError(msg)
