"""Base interface for experiment trackers."""

from abc import ABC, abstractmethod
from typing import Any

from torch import nn

from torch_batteries.tracking.types import (
    Run,
)


class ExperimentTracker(ABC):
    """
    Abstract base class for experiment tracking backends.

    Provides a unified interface for logging experiments, metrics, and artifacts
    to various tracking services (e.g. Weights & Biases).

    The tracker is a standalone service that can be used independently or
    integrated with training via ExperimentTrackingCallback.
    """

    @abstractmethod
    def init(
        self,
        run: Run,
    ) -> None:
        """
        Initialize the tracking session.

        Args:
            run: Run configuration
        """

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """
        Check if the tracker has been initialized.

        Returns:
            bool: True if initialized, False otherwise
        """

    @abstractmethod
    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
        prefix: str | None = None,
    ) -> None:
        """
        Log metrics to the tracker.

        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number (epoch, iteration, etc.)
            prefix: Optional prefix for metric names (e.g., "train/", "val/")
        """

    @abstractmethod
    def finish(self, exit_code: int = 0) -> None:
        """
        Finish tracking the experiment.

        Args:
            exit_code: Exit code (0 for success, non-zero for failure)
        """

    @abstractmethod
    def log_summary(self, summary: dict[str, Any]) -> None:
        """
        Log summary of the experiment.

        Args:
            summary: Summary metrics/info
        """

    @abstractmethod
    def log_model(
        self,
        model: nn.Module,
        name: str = "model",
        *,
        aliases: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log a trained model artifact.

        Trackers that support artifact logging (e.g. W&B) can override this.

        Args:
            model: Trained PyTorch model
            name: Artifact base name
            aliases: Optional artifact aliases (backend-specific)
            metadata: Optional artifact metadata (backend-specific)
        """
