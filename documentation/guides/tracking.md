# Experiment Tracking

Tracking is optional and backend-neutral through `ExperimentTracker`. The maintained
backend integrates Weights & Biases.

## Install W&B

```bash
python -m pip install "torch-batteries[wandb]"
```

Importing the core package does not require W&B. Constructing `WandbTracker` without
the extra installed raises an installation-focused `ImportError`.

## Configure a run

```python
from torch_batteries.callbacks import ExperimentTrackingCallback
from torch_batteries.tracking import Run, WandbTracker

tracker = WandbTracker(project="image-classification", entity=None)
run = Run(
    name="resnet-baseline",
    group="resnet-experiments",
    job_type="training",
    description="Baseline before augmentation changes",
    tags=["baseline", "resnet"],
    config={"learning_rate": 1e-3, "batch_size": 64},
)
tracking = ExperimentTrackingCallback(
    tracker,
    run,
    log_every_n_steps=10,
)

battery = Battery(model, optimizer=optimizer, callbacks=[tracking])
battery.train(train_loader, val_loader, epochs=20)
```

The callback initializes the run, logs selected train steps, logs validation metrics
at phase end, records histories and completion counters in the summary, uploads the
final model artifact, and finishes the run. Its global-step and epoch counters
participate in full checkpoints.

## Offline development

Use W&B offline mode when credentials or outbound network access are unavailable:

```bash
WANDB_MODE=offline python train.py
```

Or set it before importing W&B in a notebook:

```python
import os

os.environ.setdefault("WANDB_MODE", "offline")
```

Offline run files are written locally by W&B and can be synchronized later using its
CLI. Do not commit run directories or credentials.

## Custom backends

Implement `ExperimentTracker` to support another service. A backend must initialize a
`Run`, expose initialization state, log metrics and summaries, log a model artifact,
and finish with an exit code. It can then be passed to the same tracking callback.
