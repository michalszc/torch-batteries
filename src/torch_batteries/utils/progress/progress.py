"""Unified progress tracking for torch-batteries."""

import time
from typing import Any

from tqdm import tqdm

from .types import Phase, ProgressMetrics


class Progress:
    """
    Unified progress tracker for training, validation, testing, and prediction.

    Handles both epoch-level and batch-level progress tracking with different
    verbosity levels.

    Verbosity levels:
    - verbose=0: No output (silent)
    - verbose=1: Progress bars for batches
    - verbose=2: Simple epoch-level text output

    Args:
        verbose: Verbosity level (0, 1, or 2)
        total_epochs: Total number of epochs (for training)

    Example:
        ```python
        progress = Progress(verbose=1, total_epochs=10)

        for epoch in range(10):
            progress.start_epoch(epoch)

            # Training phase
            progress.start_phase(Phase.TRAIN, total_batches=100)
            for batch in train_loader:
                # ... training ...
                progress.update({"loss": loss.item()}, batch_size)
            train_loss = progress.end_phase()

            # Validation phase
            progress.start_phase(Phase.VALIDATION, total_batches=20)
            for batch in val_loader:
                # ... validation ...
                progress.update({"loss": loss.item()}, batch_size)
            val_loss = progress.end_phase()

            progress.end_epoch(train_loss, val_loss)

        progress.end_training()
        ```
    """

    __slots__ = (
        "_current_batch",
        "_current_epoch",
        "_current_phase",
        "_epoch_start_time",
        "_pbar",
        "_total_batches",
        "_total_epochs",
        "_total_loss",
        "_total_samples",
        "_training_start_time",
        "_verbose",
    )

    def __init__(self, verbose: int = 1, total_epochs: int = 1):
        """Initialize progress tracker.

        Args:
            verbose: Verbosity level (0=silent, 1=progress bars, 2=simple text).
            total_epochs: Total number of epochs.
        """
        if verbose not in (0, 1, 2):
            msg = f"Invalid verbose level: {verbose}. Must be 0, 1, or 2."
            raise ValueError(msg)

        self._verbose = verbose
        self._total_epochs = total_epochs
        self._current_epoch = 0
        self._current_phase: Phase | None = None
        self._current_batch = 0
        self._total_batches = 0
        self._total_loss = 0.0
        self._total_samples = 0
        self._epoch_start_time = 0.0
        self._training_start_time = time.time()
        self._pbar: Any | None = None

    def start_epoch(self, epoch: int) -> None:
        """Start a new epoch.

        Args:
            epoch: The epoch number (0-indexed).
        """
        self._current_epoch = epoch
        self._epoch_start_time = time.time()

        if self._verbose == 1:
            print(f"Epoch {epoch + 1}/{self._total_epochs}")

    def start_phase(self, phase: Phase, total_batches: int = 0) -> None:
        """Start a new phase (train, validation, test, predict).

        Args:
            phase: The training phase.
            total_batches: Total number of batches in the phase.
        """
        self._current_phase = phase
        self._total_batches = total_batches
        self._current_batch = 0
        self._total_loss = 0.0
        self._total_samples = 0
        self._pbar = None

        if self._verbose == 1 and total_batches > 0:
            phase_name = phase.value.capitalize()
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
        """Update progress after processing a batch.

        Args:
            metrics: Optional metrics dictionary containing 'loss' and other metrics.
            batch_size: Optional batch size for averaging metrics.
        """
        self._current_batch += 1

        if metrics and "loss" in metrics and batch_size is not None:
            self._total_loss += metrics["loss"] * batch_size
            self._total_samples += batch_size

        if self._verbose == 1 and self._pbar:
            if self._total_samples > 0:
                avg_loss = self._total_loss / self._total_samples
                loss_label = (
                    "val_loss" if self._current_phase == Phase.VALIDATION else "loss"
                )
                self._pbar.set_postfix_str(f"{loss_label}: {avg_loss:.4f}")
            self._pbar.update(1)

    def end_phase(self) -> float:
        """End the current phase and return average loss.

        Returns:
            Average loss across all batches, or 0.0 if no samples processed.
        """
        if self._verbose == 1 and self._pbar:
            self._pbar.close()
            self._pbar = None

        return (
            self._total_loss / self._total_samples if self._total_samples > 0 else 0.0
        )

    def end_epoch(self, train_loss: float, val_loss: float | None = None) -> None:
        """End the current epoch and display summary if verbose=2.

        Args:
            train_loss: Training loss for the epoch.
            val_loss: Optional validation loss for the epoch.
        """
        if self._verbose == 2:
            epoch_time = time.time() - self._epoch_start_time
            epoch_num = self._current_epoch + 1

            if val_loss is not None:
                print(
                    f"Epoch {epoch_num}/{self._total_epochs} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f} ({epoch_time:.2f}s)"
                )
            else:
                print(
                    f"Epoch {epoch_num}/{self._total_epochs} - "
                    f"Train Loss: {train_loss:.4f} ({epoch_time:.2f}s)"
                )

    def end_training(self) -> None:
        """End the training phase and display total time if verbose=2."""
        if self._verbose == 2:
            total_time = time.time() - self._training_start_time
            print(f"Training completed in {total_time:.2f}s")
