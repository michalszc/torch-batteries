"""Public result types for trainer workflows."""

from .predict_result import PredictResult
from .step_output import StepOutput
from .test_result import TestResult
from .train_result import TrainResult

__all__ = ["PredictResult", "StepOutput", "TestResult", "TrainResult"]
