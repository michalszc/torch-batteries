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
or persistent-worker process state. Schemas 1 and 2 remain loadable but do not contain
the reproducible RNG fields.

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

Missing directories are created. A `.pth` suffix is added only when the template has
no suffix. Static Top-K templates gain an epoch field so retained files do not
overwrite each other. Deleted rankings remove their files.

With `save_weights_only=False` (the default), each retained file is a full Battery
checkpoint. With `True`, it contains only `model.state_dict()` and can initialize a
model but cannot resume optimizer, callback, metric, epoch, or history state.

```python
inference_battery.load_checkpoint(checkpoint.best_model_path)
```

Raw weights are detected automatically. Loading them deliberately leaves resume state
disabled.

## Strict restoration failures

Restoration fails rather than silently changing an experiment when:

- The checkpoint schema is unknown.
- A saved optimizer exists but the new battery has none.
- Resumable callback types or order differ.
- Stateful metric names differ.
- The saved and configured DataPack types differ, or a required DataPack is missing.
- A callback's fixed configuration differs, such as accumulation steps or precision.
- The requested total resume target contains no new epoch.
