# Troubleshooting

## No charged step was found

Each requested workflow needs its corresponding model method:

```python
@charge(Event.TRAIN_STEP)
def training_step(self, context: EventContext) -> StepOutput:
    ...
```

Supplying a validation loader without `VALIDATION_STEP` fails before training begins.

## Optimizer is required for training

Pass an optimizer during construction or assign `battery.optimizer` before `train`.
Testing and prediction do not require one.

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
configuration, and resumable metrics. Raw weights initialize only the model and are
not a resumable checkpoint. Load checkpoint files only from trusted sources.

## W&B is unavailable

Install `torch-batteries[wandb]` in the environment running the process. Use
`WANDB_MODE=offline` when authentication or network access is intentionally absent.

## Logging is too quiet or too noisy

Training progress is controlled by `verbose`. Package diagnostics use Python logging
and default to warnings. Set `TORCH_BATTERIES_LOG_LEVEL` to `DEBUG`, `INFO`,
`WARNING`, or `ERROR` before importing the package.
