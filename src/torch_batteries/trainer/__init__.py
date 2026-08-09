"""Event-driven model training, evaluation, and prediction.

## Public API

- **`Battery`** — orchestrates device placement, steps, optimization, metrics,
  callbacks, checkpoints, testing, and prediction.
- **`StepOutput`** — explicit loss, predictions, targets, and manual metrics returned
  from a train, validation, or test step.
- **`TrainResult`** — per-epoch training and validation histories.
- **`TestResult`** — aggregate test loss and metrics.
- **`PredictResult`** — collected or recursively concatenated prediction output.
"""

from .core import Battery
from .types import PredictResult, StepOutput, TestResult, TrainResult

__all__ = ["Battery", "PredictResult", "StepOutput", "TestResult", "TrainResult"]
