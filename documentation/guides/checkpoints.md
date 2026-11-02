# Checkpoints and Resume

`torch-batteries` supports raw model weights and full resumable training state. Load
only checkpoint files from trusted sources.

## Save complete state manually

```python
battery.save_checkpoint("checkpoints/latest.pt")
```

The write is atomic and creates missing parent directories. A full checkpoint stores:

- Model and optimizer state
- Ordered resumable callback state
- Optional state exposed by configured metrics
- The qualified DataPack type and its optional `state_dict()`
- Last completed epoch and optimizer-step index
- Accumulated train and validation histories
- Python, PyTorch CPU, all available CUDA, available MPS, and optional NumPy RNG state
- Distinct generators reachable from each used DataLoader, sampler, or batch sampler

Datasets, DataLoaders, workers, and open resources are not included. NumPy state is
stored when NumPy is already available; NumPy is not a core dependency. A DataPack can
preserve construction inputs such as split indices or a streaming position in its own
state; it should not return live data objects.

## Load and continue

Recreate the same model, optimizer, callback order/configuration, and stateful metrics:

```python
model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
battery = Battery(model, optimizer=optimizer, callbacks=callbacks, metrics=metrics)
battery.load_checkpoint("checkpoints/latest.pt")
battery.fit(train_loader, val_loader, epochs=20)
```

Or load immediately before training:

```python
battery.fit(
    train_loader,
    val_loader,
    epochs=20,
    resume_from="checkpoints/latest.pt",
    resume_epochs_mode="total",
)
```

`total` treats `epochs` as the final one-based epoch target. `additional` runs that
many new epochs after the checkpoint:

```python
battery.train(
    train_loader,
    epochs=5,
    resume_from="checkpoints/latest.pt",
    resume_epochs_mode="additional",
)
```

Optimizer tensors are mapped to the battery's current device. Model loading is strict.
Callback types and order and resumable metric names must match the saved state.
If the checkpoint contains DataPack state, the same qualified DataPack type must be
attached. `fit(resume_from=...)` and `train(resume_from=...)` restore it before
`SETUP_DATA`, so saved splits affect the loaders used by the resumed workflow.

Global RNG state is restored after all other checkpoint components. Loader-generator
state is retained by phase and applied after explicit or DataPack-created loaders are
available, before resumed iteration begins. If a saved accelerator backend, NumPy, or
compatible loader generator is unavailable, restoration logs a warning and continues.

Reproducible continuation is guaranteed at completed epoch boundaries when the model,
optimizer, data, and equivalent generator-backed loaders are recreated. Checkpoints do
not reproduce an iterator from the middle of an epoch, already-prefetched worker data,
or persistent-worker process state. Full checkpoints from earlier releases cannot be
loaded directly in 1.0.0.

Full and raw-model restoration is transactional. The complete serialized structure and
component compatibility are checked first. If applying any model, optimizer, callback,
metric, DataPack, Battery, loader-generator, or RNG state fails, the pre-load state is
restored and the original exception is raised. Custom callbacks, metrics, and DataPacks
participate in this guarantee when their `state_dict()` output can be passed back to
`load_state_dict()` to restore the same state.

## Keep the best checkpoints

```python
checkpoint = ModelCheckpoint(
    phase="validation",
    metric="accuracy",
    mode="max",
    save_dir="checkpoints",
    save_path="epoch={epoch}-accuracy={accuracy:.4f}.pt",
    save_top_k=3,
    save_weights_only=False,
)
```

Both `phase` and `metric` are required. Missing directories are created. A `.pth`
suffix is added only when the template has no suffix. Static templates gain an epoch
field when retaining multiple checkpoints or using unlimited retention, so retained
files do not overwrite each other. Evicted periodic checkpoints are deleted.

### Retention and saving intervals

`save_top_k` defaults to `1`. Set it to a positive integer for ranked retention, or
`-1` to retain every eligible periodic checkpoint, including worse scores and ties.
Zero and values below `-1` are invalid. `best_model_path` and `best_score` still identify
the best retained checkpoint in unlimited mode.

`save_every_n_epochs` defaults to `1` and must be a positive integer. It uses absolute,
one-based epoch numbers: a value of `3` allows saves at epochs 3, 6, 9, and so on,
including after resume. Ineligible epochs are skipped before monitored metrics are
accessed.

For `phase="validation"`, validation must actually run on the eligible epoch. For
example, `validate_every_n_epochs=2` together with `save_every_n_epochs=3` allows
validation checkpoints at epochs 6, 12, and so on. Saving never runs extra validation.

Periodic checkpoints use `AFTER_TRAIN_EPOCH` or `AFTER_VALIDATION_EPOCH`. The latter
already has the computed validation metrics; the enclosing workflow appends that
validation history and captures validation DataLoader generator state afterward.
Use a final checkpoint when you need the state after the complete training workflow.

Each callback run also maintains a private
`.torch-batteries-top-k-<run-id>.json` ranking manifest in `save_dir`. Keep it with
the checkpoint files when continuing the same run. Its identity and monitor
configuration isolate the ranking from other callbacks and fresh runs in that
directory. Replayed epoch filenames gain a `-v1`, `-v2`, ... suffix when needed to
preserve an existing retained checkpoint.

Restoring an older checkpoint loads its model, optimizer, epoch, and history state,
then reconciles checkpoint management with the current manifest. For example, scores
`0.80, 0.79, 0.90, 0.85` with `mode="max"` and `save_top_k=2` retain epochs 3 and 4.
Loading epoch 3 still tracks both files, even though its historical callback snapshot
referenced the now-deleted epoch 1. Loading does not modify checkpoint files or the
manifest.

Missing sibling files no longer invalidate an otherwise valid checkpoint. If the
manifest is missing, invalid, or absent from a legacy callback snapshot, restoration
warns and retains only existing files from that snapshot. Newer directory entries
cannot be recovered automatically in this fallback. Structurally corrupt callback
state and incompatible monitor configurations are still rejected.

Checkpoint and manifest writes complete before displaced files are removed. A failed
write restores the previous ranking and checkpoint contents. One writer must own a
ranking; concurrent writers and atomic recovery across multiple files after a process
crash are not supported.

With `save_weights_only=False` (the default), each retained file is a full Battery
checkpoint. With `True`, it contains only `model.state_dict()` and can initialize a
model but cannot resume optimizer, callback, metric, epoch, or history state.

```python
inference_battery.load_checkpoint(checkpoint.best_model_path)
```

Raw weights are detected automatically. Loading them deliberately leaves resume state
disabled.

## Save the final training state

Keep every second epoch and an additional final checkpoint:

```python
checkpoint = ModelCheckpoint(
    phase="validation",
    metric="accuracy",
    save_dir="checkpoints",
    save_path="models/{epoch}.pt",
    save_top_k=-1,
    save_every_n_epochs=2,
    save_final=True,
)
battery = Battery(model, optimizer=optimizer, callbacks=[checkpoint])
battery.fit(train_loader, val_loader, epochs=5)

assert checkpoint.final_model_path is not None
battery.load_checkpoint(checkpoint.final_model_path)
```

`save_final` defaults to `False`. When enabled, it saves the current model at
`AFTER_TRAIN` after `train()` or `fit()`, for either monitoring phase, including
normal early stopping. It ignores the periodic save interval and top-k qualification.
It does not run extra validation or create an additional final file after standalone
`validate()`. A failure during training or validation does not reach the final-save
event.

The final file respects `save_weights_only`: the default saves complete Battery
state; `True` saves raw model weights. Full final checkpoints contain completed epoch,
history, optimizer, and stopping state. Handlers run in callback order, so place a
callback that restores model weights before `ModelCheckpoint` if those restored
weights should be included in the final file.

`final_save_path` optionally overrides the final filename template. By default, the
filename portion of `save_path` gains a `final-` prefix, preserving its parent
directory:

| `save_path` | Default final filename |
| --- | --- |
| Omitted | `final-epoch=5.pth` at epoch 5 |
| `best.pt` | `final-best.pt` |
| `models/{epoch}.pt` | `models/final-epoch=5.pt` at epoch 5 |

For example, `final_save_path="exports/last.pt"` writes the final file there within
`save_dir`. The same template fields, metric-name insertion, and suffix rules apply
to periodic and final files. Final template metrics come from the selected phase's
metrics available at `AFTER_TRAIN`; validation metrics may be from an earlier epoch
when the last epoch skips validation. Missing template fields raise a formatting
error. A filename using only `{epoch}` needs no monitored metric at completion.

The read-only `final_model_path` is `None` until a successful final save, then reports
the actual path, including any collision suffix. Each callback/run identity retains
only its latest final checkpoint. This file is tracked in the manifest separately
from `best_k_models` and never changes best/kth scores. Its replacement is committed
before the previous owned final file is removed; periodic files and unrelated files
are protected by collision suffixes. Full restoration reconciles both the periodic
ranking and the latest final path with the manifest.

Older callback metadata without the new options defaults to
`save_every_n_epochs=1`, `save_final=False`, and `final_save_path=None`, with no final
path recorded. Recreate matching saving options when resuming a configured run.

## Strict restoration failures

Restoration fails rather than silently changing an experiment when:

- The checkpoint format is unsupported.
- A saved optimizer exists but the new battery has none.
- Resumable callback types or order differ.
- Stateful metric names differ.
- The saved and configured DataPack types differ, or a required DataPack is missing.
- A callback's fixed configuration differs, such as accumulation steps or precision.
- A checkpoint callback's saved monitor, filename template, or retention policy differs.
- The requested total resume target contains no new epoch.
