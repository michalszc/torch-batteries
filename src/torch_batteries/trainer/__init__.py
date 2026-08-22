"""Event-driven model training, evaluation, and prediction.

## Public API

- **`Battery`** — orchestrates device placement, steps, optimization, metrics,
  callbacks, checkpoints, testing, and prediction.
- **`StepOutput`** — explicit loss, predictions, targets, and manual metrics returned
  from a train, validation, or test step.
- **`FitResult`** — per-epoch training and optional validation histories.
- **`TrainResult`** — per-epoch training histories with temporary validation
  compatibility fields.
- **`ValidationResult`** — aggregate standalone validation loss and metrics.
- **`TestResult`** — aggregate test loss and metrics.
- **`PredictResult`** — collected or recursively concatenated prediction output.
"""

from .core import Battery
from .types import (
    FitResult,
    PredictResult,
    StepOutput,
    TestResult,
    TrainResult,
    ValidationResult,
)

__all__ = [
    "Battery",
    "FitResult",
    "PredictResult",
    "StepOutput",
    "TestResult",
    "TrainResult",
    "ValidationResult",
]
