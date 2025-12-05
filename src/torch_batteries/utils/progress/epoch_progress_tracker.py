"""Progress tracker for individual epoch phases."""

from typing import Any

from tqdm import tqdm

from .types import Phase, ProgressMetrics


class EpochProgressTracker:
    """
    Simple progress tracker for individual phases within an epoch.

    Shows progress bars for training, validation, testing, or prediction.
    """

    __slots__ = (
        "_current_batch",
        "_pbar",
        "_phase",
        "_total_batches",
        "_total_loss",
        "_total_samples",
        "_verbose",
    )

    def __init__(
        self, verbose: int = 1, phase: Phase = Phase.TRAIN, total_batches: int = 0
    ):
        self._verbose = verbose
        self._phase = phase
        self._total_batches = total_batches

        self._current_batch = 0
        self._total_loss = 0.0
        self._total_samples = 0

        self._pbar: Any | None = None

        if self._verbose == 1 and total_batches > 0:
            phase_name = self._phase.value.capitalize()
            self._pbar = tqdm(
                total=total_batches,
                desc=phase_name,
                ncols=80,
                bar_format="{desc}: {n}/{total} {bar} {percentage:3.0f}%{postfix}",
                leave=True,
            )

    def update(
        self, metrics: ProgressMetrics | None = None, batch_size: int | None = None
    ) -> None:
        """Update progress after processing a batch."""
        self._current_batch += 1

        # Only update loss tracking if metrics and loss are provided
        if metrics and "loss" in metrics and batch_size is not None:
            self._total_loss += metrics["loss"] * batch_size
            self._total_samples += batch_size

        if self._pbar:
            # Only show loss if we're tracking it
            if self._total_samples > 0:
                avg_loss = self._total_loss / self._total_samples
                if self._phase == Phase.VALIDATION:
                    self._pbar.set_postfix_str(f"val_loss: {avg_loss:.4f}")
                else:
                    self._pbar.set_postfix_str(f"loss: {avg_loss:.4f}")
            self._pbar.update(1)

    def close(self) -> float:
        """Close the progress tracker and return average loss."""
        if self._pbar:
            self._pbar.close()

        return (
            self._total_loss / self._total_samples if self._total_samples > 0 else 0.0
        )
