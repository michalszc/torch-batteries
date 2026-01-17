"""
torch-batteries: A lightweight Python package for PyTorch workflow abstractions.
"""

__version__ = "0.5.0"
__author__ = ["Michal Szczygiel", "Arkadiusz Paterak", "Antoni Zięciak"]

# Import main components
from .events import Event, EventContext, charge
from .tracking import (
    Run,
)
from .trainer import Battery, PredictResult, TestResult, TrainResult

__all__ = [
    "Battery",
    "Event",
    "EventContext",
    "PredictResult",
    "Run",
    "TestResult",
    "TrainResult",
    "charge",
]
