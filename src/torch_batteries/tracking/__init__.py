"""Backend-neutral experiment tracking with optional W&B integration.

## Public API

- **`ExperimentTracker`** — interface implemented by tracking backends.
- **`Run`** — immutable run name, grouping, tags, description, and configuration.
- **`WandbTracker`** — optional Weights & Biases backend installed with the
  ``wandb`` extra.
"""

from .base import ExperimentTracker
from .types import (
    Run,
)
from .wandb import WandbTracker

__all__ = [
    "ExperimentTracker",
    "Run",
    "WandbTracker",
]
