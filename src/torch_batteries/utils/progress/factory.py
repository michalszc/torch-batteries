"""Factory for creating progress trackers."""

from torch_batteries.utils.logging import get_logger

from .base import Progress
from .progress_bar import BarProgress
from .silent import SilentProgress
from .simple import SimpleProgress

logger = get_logger("progress.factory")


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
                progress: Progress = SilentProgress(total_epochs)
            case 1:
                progress = BarProgress(total_epochs)
            case 2:
                progress = SimpleProgress(total_epochs)
            case _:
                msg = f"Invalid verbose level: {verbose}. Must be 0, 1, or 2."
                raise ValueError(msg)
        logger.debug(
            "Created %s for verbose level %d", type(progress).__name__, verbose
        )
        return progress
