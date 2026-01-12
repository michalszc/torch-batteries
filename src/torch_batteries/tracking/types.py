"""Type definitions for experiment tracking."""

from dataclasses import dataclass, field
from typing import Any


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
    description: str | None = None


@dataclass
class Experiment:
    """
    An investigation or hypothesis within a project.

    Groups related runs that answer a specific research question.
    For example: "Effect of learning rate on convergence"

    Args:
        name: Experiment identifier (e.g., "lr-sweep")
        description: What question is being answered
    """

    name: str
    description: str | None = None


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
        description: Optional description of the run
    """

    name: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    group: str | None = None
    job_type: str | None = "train"
    description: str | None = None
