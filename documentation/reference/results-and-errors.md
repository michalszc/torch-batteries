# Results and Exceptions

## Fit result

`Battery.fit` returns `FitResult` with these keys:

| Key | Value |
| --- | --- |
| `train_loss` | Average loss for each completed train epoch |
| `val_loss` | Average loss for each completed validation phase |
| `train_metrics` | Metric name to per-epoch history |
| `val_metrics` | Metric name to per-epoch history |
| `epochs_completed` | Last completed absolute epoch, including resumed work |
| `optimizer_steps` | Cumulative optimizer updates |
| `stopped_early` | Whether an orderly stop was requested |
| `stop_reason` | Human-readable reason or `None` |

Validation collections are empty when no validation loader is supplied. Early
stopping returns the histories completed before the stop flag was observed. With
validation cadence greater than one, validation histories contain only the epochs
that were validated.

## Training result

`Battery.train` returns `TrainResult` with `train_loss`, `train_metrics`, and the same
four execution metadata fields as `FitResult`. It does not run validation.

## Validation result

`Battery.validate` returns `ValidationResult`. `val_loss` is always present and
`val_metrics` is included when at least one metric beyond loss was produced. Unlike
`fit()`, standalone validation requires validation data.

## Test result

`test_loss` is always present. `test_metrics` is included when at least one metric
beyond loss was produced. With several named datasets, the loss and plain metric keys
are sample-weighted aggregates. Dataset-specific metrics use `dataset:metric` names,
including `dataset:loss`. With one dataset, only plain names appear.

## Prediction result

`predictions` is a list of batch outputs by default. With `concatenate=True`, it is a
recursively concatenated tensor/dictionary/tuple/list structure instead. When several
named prediction datasets run, `predictions` maps dataset names to their output
structures; a selected or single dataset has the ordinary output structure.

## Common validation errors

| Error | Meaning |
| --- | --- |
| `Optimizer is required for training` | Construct or assign an optimizer before `fit` or `train` |
| `... loader must not be empty` | The selected loader reports zero batches |
| `No method decorated with ... found` | Add the charged step required by the workflow |
| `... loss must be a scalar tensor` | Reduce the batch loss before returning it |
| `... must return StepOutput ... when Battery metrics are configured` | Include predictions and targets in the step result |
| `Prediction structures differ across batches` | Return the same nested output shape from every prediction batch |

Checkpoint format, callback-order, and metric-state mismatches intentionally fail
strict restoration rather than silently resuming a different experiment.

Incorrect runtime argument types raise `TypeError`; unsupported values and conflicting
configuration raise `ValueError`; invalid workflow state raises `RuntimeError`.
Filesystem failures retain their underlying `OSError` cause. Error messages identify
the component that owns the contract—Battery, DataPack, event dispatch, callback,
tracking backend, or utility—so callers do not need to infer the source from an
unrelated subsystem.
