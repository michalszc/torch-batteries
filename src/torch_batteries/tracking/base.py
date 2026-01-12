"""Base interface for experiment trackers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from torch import nn

from torch_batteries.tracking.types import (
    Artifact,
    ArtifactMaterialization,
    Experiment,
    Project,
    Run,
)


class ExperimentTracker(ABC):
    """
    Abstract base class for experiment tracking backends.

    Provides a unified interface for logging experiments, metrics, and artifacts
    to various tracking services (wandb, MLflow, TensorBoard, etc.).

    The tracker is a standalone service that can be used independently or
    integrated with training via the ExperimentTrackingCallback.

    Example:
        ```python
        # Standalone usage
        tracker = WandbTracker()
        tracker.init(
            project=Project("my-research"),
            experiment=Experiment("lr-study"),
            run=Run(config={"lr": 0.001})
        )

        # Log during training
        tracker.log_metrics({"loss": 0.5, "accuracy": 0.92}, step=100)

        # Save artifacts
        tracker.save_artifact(
            Asset("model", AssetType.MODEL, "model.pth")
        )

        tracker.finish()
        ```
    """

    @abstractmethod
    def init(
        self,
        project: Project,
        experiment: Experiment | None = None,
        run: Run | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the tracking session.

        Args:
            project: The project this run belongs to
            experiment: Optional experiment grouping
            run: Run configuration
            **kwargs: Backend-specific configuration
        """
        pass

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
        pass

    @abstractmethod
    def log_config(self, config: dict[str, Any]) -> None:
        """
        Log configuration/hyperparameters.

        Args:
            config: Configuration dictionary to log
        """
        pass

    @abstractmethod
    def save_artifact(
        self,
        artifact: Artifact,
        dependencies: list[str] | None = None,
    ) -> ArtifactMaterialization:
        """
        Save an artifact and track its lineage.

        Args:
            artifact: The artifact to save
            dependencies: Names of artifacts this depends on

        Returns:
            ArtifactMaterialization record
        """
        pass

    @abstractmethod
    def finish(self, exit_code: int = 0) -> None:
        """
        Finish the tracking session.

        Args:
            exit_code: Exit code (0 for success, non-zero for failure)
        """
        pass

    @abstractmethod
    def log_summary(self, summary: dict[str, Any]) -> None:
        """
        Log summary statistics that appear at the top level.

        Args:
            summary: Summary metrics/info
        """
        pass

    def log_file(
        self,
        file_path: str | Path,
        name: str | None = None,
        base_path: str | Path | None = None,
    ) -> None:
        """
        Log a file as an artifact.

        Args:
            file_path: Path to the file
            name: Optional name for the file in tracking system
            base_path: Optional base path for relative paths
        """
        # Default implementation - can be overridden
        file_path = Path(file_path)
        name = name or file_path.name

        # Create artifact and save it
        artifact = Artifact(
            name=name,
            type=self._infer_artifact_type(file_path),
            path=file_path,
        )
        self.save_artifact(artifact)

    def _infer_artifact_type(self, path: Path) -> Any:
        """
        Infer artifact type from file extension.

        Args:
            path: File path

        Returns:
            ArtifactType enum value
        """
        from torch_batteries.tracking.types import ArtifactType

        suffix = path.suffix.lower()
        if suffix in {".pth", ".pt", ".ckpt", ".h5"}:
            return ArtifactType.MODEL
        if suffix in {".png", ".jpg", ".jpeg", ".svg"}:
            return ArtifactType.PLOT
        if suffix in {".csv", ".json", ".jsonl"}:
            return ArtifactType.METRICS
        return ArtifactType.CONFIG
