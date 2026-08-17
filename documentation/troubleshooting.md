# Troubleshooting

## No charged step was found

Each requested workflow needs its corresponding model method:

```python
@charge(Event.TRAIN_STEP)
def training_step(self, context: EventContext) -> StepOutput:
    ...
```

Supplying validation data without `VALIDATION_STEP` fails before fitting or standalone
validation begins.

## No loader or DataPack dataset was provided

Pass an explicit primary DataLoader, or attach `Battery(data_pack=...)` with a
`SETUP_DATA` handler returning `DatasetBundle`. An implicit fit/train, test, or
prediction workflow requires its corresponding dataset. Validation is optional for
`fit()` and compatibility `train()`, but required for standalone `validate()`.

Do not pass only `val_loader`: an explicit train loader selects direct-loader mode,
and Battery never mixes it with implicit DataPack loaders.

## A DataLoaderConfig is invalid

Do not combine a sampler with `shuffle=True`. A batch sampler requires
`batch_size=None` and cannot be combined with shuffle, a sampler, or `drop_last`.
Prefetching and persistent workers require `num_workers > 0`; iterable datasets
cannot be shuffled.

## Optimizer is required for training

Pass an optimizer during construction or assign `battery.optimizer` before `fit` or
`train`. Standalone validation, testing, and prediction do not require one.

## Automatic metrics require predictions and targets

When `Battery(metrics=...)` is configured, return `StepOutput` with values produced by
the same forward pass:

```python
return StepOutput(loss=loss, predictions=outputs, targets=targets)
```

## A monitored metric is unavailable

Use exactly the same metric name in `Battery(metrics=...)`, manual step metrics,
`EarlyStopping`, `ModelCheckpoint`, and plateau schedulers. `"loss"` is included in
every completed train, validation, and test phase.

## CUDA or MPS runs out of memory

- Reduce loader batch size.
- Use `GradientAccumulation` to preserve a larger effective batch.
- Use `MixedPrecision` where the model and device support it.
- Stream prediction with `predict_iter`.
- Set `move_to_cpu=True` when collecting outputs.
- Avoid retaining GPU tensors in lifecycle callbacks.
- Remember that `CollectedMetric` stores all predictions and targets on CPU.

## Prediction cannot concatenate outputs

Every batch must return matching container structure and compatible tensor shapes.
Leave `concatenate=False` for variable-length or non-tensor outputs and combine them in
application code.

## A checkpoint cannot be resumed

Recreate the same model, optimizer, ordered `Callback` subclasses, callback
configuration, resumable metrics, and DataPack type. Raw weights initialize only the
model and are not a resumable checkpoint. Load checkpoint files only from trusted
sources.

## W&B is unavailable

Install `torch-batteries[wandb]` in the environment running the process. Use
`WANDB_MODE=offline` when authentication or network access is intentionally absent.

## Logging is too quiet or too noisy

Training progress is controlled by `verbose`. Package diagnostics use Python logging
and default to warnings. Set `TORCH_BATTERIES_LOG_LEVEL` to `DEBUG`, `INFO`,
`WARNING`, or `ERROR` before importing the package.

Logger names follow the source module below the `torch_batteries` hierarchy, such as
`torch_batteries.trainer.core`, `torch_batteries.data.loader`, and
`torch_batteries.callbacks.early_stopping`. Applications can therefore filter one
component with ordinary Python logging configuration. Debug output describes detailed
workflow transitions and resolved configuration; info output reports user-relevant
lifecycle operations; warnings identify deprecations and recoverable fallbacks; errors
identify operations that are about to raise.
