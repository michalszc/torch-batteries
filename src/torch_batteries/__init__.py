"""
torch-batteries: A lightweight Python package for PyTorch workflow abstractions.
"""

__version__ = "0.4.1"
__author__ = ["Michal Szczygiel", "Arkadiusz Paterak", "Antoni Zięciak"]

# Import main components
from .events import Event, EventContext, charge
from .trainer import Battery, PredictResult, TestResult, TrainResult

__all__ = [
    "Battery",
    "Event",
    "EventContext",
    "PredictResult",
    "TestResult",
    "TrainResult",
    "charge",
]
