"""Reusable callbacks for workflow control and optimization.

## Callback contract

- **`Callback`** — optional resumable-state base class for custom callbacks.

## Training control

- **`EarlyStopping`** — stops training when a monitored metric stops improving.
- **`ModelCheckpoint`** — retains the best weights-only or full-state checkpoints.
- **`ExperimentTrackingCallback`** — logs training lifecycle data through a tracker.

## Optimization

- **`GradientAccumulation`** — groups batches into fewer optimizer steps.
- **`GradientClip`** — applies value- or norm-based gradient clipping.
- **`MixedPrecision`** — supplies autocast, scaled backward, and optimizer stepping.
- **`LearningRateScheduler`** — advances step-, epoch-, or validation-based schedulers.
"""

from .base import Callback
from .early_stopping import EarlyStopping
from .experiment_tracking import ExperimentTrackingCallback
from .gradient_accumulation import GradientAccumulation
from .gradient_clip import GradientClip
from .learning_rate_scheduler import LearningRateScheduler
from .mixed_precision import MixedPrecision
from .model_checkpoint import ModelCheckpoint

__all__ = [
    "Callback",
    "EarlyStopping",
    "ExperimentTrackingCallback",
    "GradientAccumulation",
    "GradientClip",
    "LearningRateScheduler",
    "MixedPrecision",
    "ModelCheckpoint",
]
