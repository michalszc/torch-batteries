# Experiment Tracking

Tracking is optional and backend-neutral through `ExperimentTracker`. `LocalTracker`
stores runs on disk without an account. `WandbTracker` integrates Weights & Biases.

## Track locally

```python
from torch_batteries.callbacks import ExperimentTrackingCallback
from torch_batteries.tracking import LocalTracker, Run

tracker = LocalTracker("model_a", save_dir="my_experiments")
tracking = ExperimentTrackingCallback(
    tracker,
    run=Run(config={"batch_size": 64, "learning_rate": 1e-3}),
)
battery = Battery(model, optimizer=optimizer, callbacks=[tracking])
battery.fit(train_loader, val_loader, epochs=20)
```

Each run allocates `my_experiments/model_a/version_N/`. `hparams.yaml` records the
model architecture, parameter count, optimizer settings, and any overriding
`Run.config` values. `metrics.csv` has one row per completed epoch and dynamically
named columns such as `train/loss`, `train/accuracy`, and `val/accuracy`. Validation
values merge into their matching epoch row; skipped validation epochs have empty
validation columns. `summary.yaml` contains completion totals and only the latest
value of each metric. The local backend does not save model artifacts.

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
battery.fit(train_loader, val_loader, epochs=20)
```

For W&B, the callback initializes the run, logs selected train steps, logs validation
metrics at phase end, records histories and completion counters in the summary,
uploads the final model artifact, and finishes the run. For `LocalTracker`, the same
callback logs completed epoch totals and a latest-value summary. Its global-step and
epoch counters participate in full checkpoints.

If an exception escapes a Battery workflow after the tracker initializes,
``ON_EXCEPTION`` finishes the run with ``exit_code=1``. Failure cleanup does not upload
a final model artifact. Exceptions raised by cleanup handlers are logged without
replacing the original workflow failure.

## W&B offline development

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
