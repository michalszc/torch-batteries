# API Reference

This reference is generated from the installed package's public docstrings by
mkdocstrings. It is the authoritative source for signatures, accepted arguments,
return contracts, event context fields, and documented exceptions. For task-oriented
explanations, start with the [Guides](../guides/index.md).

## Reference map

| Area | Page | Main public objects |
| --- | --- | --- |
| Workflows | [Trainer API](trainer.md) | `Battery`, `StepOutput`, `TrainResult`, `TestResult`, and `PredictResult` |
| Data construction | [Data API](data.md) | `DataPack`, `DatasetBundle`, `DataLoaderConfig`, and `DataContext` |
| Lifecycle | [Events API](events.md) | `Event`, `EventContext`, `OptimizationStep`, and `charge` |
| Extensions | [Callbacks API](callbacks.md) | `Callback`, early stopping, checkpoints, optimization callbacks, and experiment tracking |
| Measurement | [Metrics API](metrics.md) | `StatefulMetric` and `CollectedMetric` |
| Tracking | [Tracking API](tracking.md) | `ExperimentTracker`, `Run`, and `WandbTracker` |
| Schemas and failures | [Results and Exceptions](results-and-errors.md) | Result fields, epoch numbering, validation behavior, and common exceptions |

## Prefer public import paths

Examples and applications should import from the package root or a documented public
sub-package:

```python
from torch_batteries import Battery, DataPack, Event, EventContext, StepOutput, charge
from torch_batteries.callbacks import EarlyStopping, ModelCheckpoint
from torch_batteries.tracking import Run, WandbTracker
```

Do not depend on implementation modules such as `torch_batteries.trainer.core` or
individual callback files. Public imports keep user code independent of internal file
organization.

## How to use the generated pages

- Start with a class entry to confirm constructor arguments and properties.
- Check method entries for exact return structures and raised exceptions.
- For charged methods and callbacks, cross-reference the Events API to see when an
  event runs and which context keys are available.
- For training, testing, and prediction, cross-reference Results and Exceptions before
  persisting or serializing returned values.

!!! note "Optional integrations"

    The tracking API can be documented and imported without W&B installed.
    Constructing `WandbTracker` requires the `wandb` optional extra.
