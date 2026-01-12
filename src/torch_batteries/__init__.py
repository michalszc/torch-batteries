"""
torch-batteries: A lightweight Python package for PyTorch workflow abstractions.
"""

__version__ = "0.4.0"
__author__ = ["Michal Szczygiel", "Arkadiusz Paterak", "Antoni Zięciak"]

# Import main components
from .events import Event, EventContext, charge
from .trainer import Battery, PredictResult, TestResult, TrainResult
from .tracking import (
    Artifact,
    ArtifactMaterialization,
    ArtifactType,
    Experiment,
    Project,
    Run,
)

__all__ = [
    "Artifact",
    "ArtifactMaterialization",
    "ArtifactType",
    "Battery",
    "Event",
    "EventContext",
    "Experiment",
    "PredictResult",
    "Project",
    "Run",
    "TestResult",
    "TrainResult",
    "charge",
]
