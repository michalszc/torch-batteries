"""Callbacks module for torch-batteries.

Provides callback classes for training workflow control:
- **EarlyStopping**: Stop training when monitored metric stops improving
- **ModelCheckpoint**: Save model checkpoints when monitored metric improves
- **ExperimentTrackingCallback**: Track experiments using external tools
"""

from .early_stopping import EarlyStopping
from .experiment_tracking import ExperimentTrackingCallback
from .model_checkpoint import ModelCheckpoint

__all__ = [
    "EarlyStopping",
    "ExperimentTrackingCallback",
    "ModelCheckpoint",
]
