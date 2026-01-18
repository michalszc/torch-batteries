"""Callbacks module for torch-batteries.

Provides callback classes for training workflow control:
- **EarlyStopping**: Stop training when monitored metric stops improving
- **ModelCheckpoint**: Save model checkpoints when monitored metric improves
"""

from .early_stopping import EarlyStopping
from .model_checkpoint import ModelCheckpoint

__all__ = ["EarlyStopping", "ModelCheckpoint"]
