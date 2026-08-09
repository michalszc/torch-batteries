# Results and Exceptions

## Training result

`Battery.train` always returns these four keys:

| Key | Value |
| --- | --- |
| `train_loss` | Average loss for each completed train epoch |
| `val_loss` | Average loss for each completed validation phase |
| `train_metrics` | Metric name to per-epoch history |
| `val_metrics` | Metric name to per-epoch history |

Validation collections are empty when no validation loader is supplied. Early
stopping returns the histories completed before the stop flag was observed.

## Test result

`test_loss` is always present. `test_metrics` is included when at least one metric
beyond loss was produced.

## Prediction result

`predictions` is a list of batch outputs by default. With `concatenate=True`, it is a
recursively concatenated tensor/dictionary/tuple/list structure instead.

## Common validation errors

| Error | Meaning |
| --- | --- |
| `Optimizer is required for training` | Construct or assign an optimizer before `train` |
| `... loader must not be empty` | The selected loader reports zero batches |
| `No method decorated with ... found` | Add the charged step required by the workflow |
| `... loss must be a scalar tensor` | Reduce the batch loss before returning it |
| `... must return StepOutput ... when Battery metrics are configured` | Include predictions and targets in the step result |
| `Prediction structures differ across batches` | Return the same nested output shape from every prediction batch |

Checkpoint schema, callback-order, and metric-state mismatches intentionally fail
strict restoration rather than silently resuming a different experiment.
