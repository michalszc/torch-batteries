"""Trainer module for torch-batteries package.

Provides the main Battery trainer class for PyTorch model training,
testing, and prediction with event-driven workflow control.

Main Classes:
- Battery: Main trainer class with event-driven architecture
- TrainResult: Return type for training results
- TestResult: Return type for test results
- PredictResult: Return type for prediction results
- StepOutput: Explicit output from train, validation, and test steps
"""

from .core import Battery
from .types import PredictResult, StepOutput, TestResult, TrainResult

__all__ = ["Battery", "PredictResult", "StepOutput", "TestResult", "TrainResult"]
