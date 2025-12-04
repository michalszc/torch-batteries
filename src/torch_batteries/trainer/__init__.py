"""Trainer module for torch-batteries package.

This module provides the Battery trainer class and associated types.
"""

from .core import Battery
from .types import PredictResult, TestResult, TrainResult

__all__ = ["Battery", "PredictResult", "TestResult", "TrainResult"]
