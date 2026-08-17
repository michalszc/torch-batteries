"""torch-batteries: A lightweight Python package for PyTorch workflow abstractions.

torch-batteries provides a flexible training framework for PyTorch that uses
event-based decorators to define training, validation, testing, and prediction logic.

## Key Features

- Event-driven architecture with `@charge` decorator
- Automatic metric tracking and progress display
- Built-in callbacks (EarlyStopping, ModelCheckpoint)
- Flexible progress tracking (silent, progress bars, detailed logs)
- Simple API with minimal boilerplate

## Quick Start

```python
from torch_batteries import Battery, Event, StepOutput, charge
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

    @charge(Event.TRAIN_STEP)
    def training_step(self, context):
        batch = context['batch']
        x, y = batch
        pred = self(x)
        return StepOutput(
            loss=F.mse_loss(pred, y),
            predictions=pred,
            targets=y,
        )

battery = Battery(model, optimizer=optimizer) # Auto-detects device
battery.train(train_loader, val_loader, epochs=10)
```
"""

__version__ = "0.10.1"
__author__ = ["Michal Szczygiel", "Arkadiusz Paterak", "Antoni Zięciak"]

# Import main components
from .data import (
    DataContext,
    DataLoaderBundle,
    DataLoaderConfig,
    DataPack,
    DataPackHandler,
    DatasetBundle,
    ResolvedData,
)
from .events import Event, EventContext, OptimizationStep, charge
from .tracking import (
    Run,
)
from .trainer import (
    Battery,
    FitResult,
    PredictResult,
    StepOutput,
    TestResult,
    TrainResult,
    ValidationResult,
)
from .utils.metrics import CollectedMetric, StatefulMetric

__all__ = [
    "Battery",
    "CollectedMetric",
    "DataContext",
    "DataLoaderBundle",
    "DataLoaderConfig",
    "DataPack",
    "DataPackHandler",
    "DatasetBundle",
    "Event",
    "EventContext",
    "FitResult",
    "OptimizationStep",
    "PredictResult",
    "ResolvedData",
    "Run",
    "StatefulMetric",
    "StepOutput",
    "TestResult",
    "TrainResult",
    "ValidationResult",
    "callbacks",
    "charge",
    "events",
    "tracking",
    "trainer",
    "utils",
]
