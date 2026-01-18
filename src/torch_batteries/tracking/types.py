"""Type definitions for experiment tracking."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Run:
    """
    A single run with a configuration.

    Args:
        name: Optional run identifier
        group: Optional group name for organizing runs
        job_type: Optional type of job (e.g., "training", "evaluation")
        description: Optional description of the run
        config: Run-specific configuration
    """

    name: str | None = None
    group: str | None = None
    job_type: str | None = None
    description: str | None = None
    tags: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
