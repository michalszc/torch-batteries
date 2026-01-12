"""Type definitions for experiment tracking."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class ArtifactType(Enum):
    """Types of artifacts that can be tracked."""

    # Data
    DATASET = "dataset"
    PREPROCESSED_DATA = "preprocessed_data"

    # Models
    MODEL = "model"
    CHECKPOINT = "checkpoint"

    # Results
    METRICS = "metrics"
    PREDICTIONS = "predictions"

    # Artifacts
    PLOT = "plot"
    CONFIG = "config"


@dataclass
class Project:
    """
    Top-level container for research work.

    Contains multiple experiments investigating different aspects of the research.

    Args:
        name: Project identifier (e.g., "mnist-research")
        notes: Optional project description
    """

    name: str
    notes: str | None = None


@dataclass
class Experiment:
    """
    An investigation or hypothesis within a project.

    Groups related runs that answer a specific research question.
    For example: "Effect of learning rate on convergence"

    Args:
        name: Experiment identifier (e.g., "lr-sensitivity")
        description: What question is being answered
        base_config: Configuration shared across all runs
        tags: Tags for categorization
    """

    name: str
    description: str | None = None
    base_config: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


@dataclass
class Run:
    """
    A single execution with specific configuration.

    One training loop from start to finish with specific hyperparameters.

    Args:
        name: Optional run identifier (auto-generated if not provided)
        config: Run-specific configuration/hyperparameters
        group: Optional group name for organizing runs
        job_type: Type of job (e.g., "train", "eval", "inference")
        tags: Tags for this specific run
        notes: Additional notes about this run
    """

    name: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    group: str | None = None
    job_type: str | None = "train"
    tags: list[str] = field(default_factory=list)
    notes: str | None = None


@dataclass
class Artifact:
    """
    A versioned artifact produced or consumed by an experiment.

    Inspired by Dagster's asset model for tracking data lineage.

    Args:
        name: Artifact identifier
        type: Type of artifact (from ArtifactType enum)
        path: Path to the artifact file
        metadata: Additional metadata about the artifact
        created_at: When the artifact was created
        version: Artifact version identifier
        tags: Tags for categorization
    """

    name: str
    type: ArtifactType
    path: str | Path
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    version: str | None = None
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Convert path to Path object and set created_at if not provided."""
        if isinstance(self.path, str):
            self.path = Path(self.path)
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class ArtifactMaterialization:
    """
    Record of when and how an artifact was created.

    Tracks artifact lineage and dependencies for reproducibility.

    Args:
        artifact: The artifact that was created
        run_id: ID of the run that created this artifact
        timestamp: When the artifact was created
        metadata: Additional metadata about creation
        dependencies: Names of input artifacts used to create this
    """

    artifact: Artifact
    run_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
