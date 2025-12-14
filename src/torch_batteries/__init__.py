"""
torch-batteries: A lightweight Python package for PyTorch workflow abstractions.
"""

__version__ = "0.2.1"
__author__ = ["Michal Szczygiel", "Arkadiusz Paterak", "Antoni Zięciak"]

# Import main components
from .events import Event, charge
from .trainer import Battery

__all__ = ["Battery", "Event", "charge"]
