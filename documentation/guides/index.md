# Guides

These guides explain how the public pieces of torch-batteries work together in real
training and inference workflows. They focus on decisions, trade-offs, failure modes,
and complete behavior; use the [API Reference](../reference/index.md) when you need an
exact signature or event context.

## Choose a guide by goal

| Goal | Guide | Key topics |
| --- | --- | --- |
| Define phase behavior | [Training and Evaluation](training.md) | `StepOutput`, train/validation/test contracts, manual metrics, empty loaders, and result histories |
| Calculate metrics correctly | [Metrics](metrics.md) | Callable metrics, `StatefulMetric`, `CollectedMetric`, weighting, state, and memory cost |
| Use structured inputs | [Batches and Devices](batches-and-devices.md) | Tensor, tuple, dictionary, nested batches, multiple inputs, and automatic device transfer |
| Control optimization | [Callbacks and Optimization](callbacks.md) | Callback order, early stopping, accumulation, clipping, mixed precision, and schedulers |
| Preserve or resume work | [Checkpoints and Resume](checkpoints.md) | Manual saves, Top-K selection, weights-only files, full state, and resume modes |
| Run inference | [Prediction](prediction.md) | Batch preservation, recursive CPU transfer, concatenation, and `predict_iter` streaming |
| Track experiments | [Experiment Tracking](tracking.md) | W&B configuration, offline operation, automatic logging, and custom backends |

## Suggested paths

### First supervised project

Read Training and Evaluation, Metrics, and Batches and Devices. Add callbacks only
after the basic train/validation/test workflow returns the results you expect.

### Long-running or resumable training

Read Callbacks and Optimization before Checkpoints and Resume. Full checkpoints
restore optimizer, callback, metric, epoch, and history state, so callback ordering
and configuration are part of the experiment contract.

### Memory-sensitive inference

Read Prediction and prefer `predict_iter(..., move_to_cpu=True)` when retaining every
batch would be expensive. Use recursive concatenation only when all batches return the
same compatible structure.

### Tracked experiments

Read Experiment Tracking after your metrics and callbacks are stable. W&B is optional,
supports offline operation, and is not required by the core package.

## Where to look when behavior is unexpected

- Use [Troubleshooting](../troubleshooting.md) for common configuration and runtime
  failures.
- Use [Events API](../reference/events.md) for exact event frequency and context keys.
- Use [Results and Exceptions](../reference/results-and-errors.md) for returned schemas
  and public validation errors.
- Use the maintained [Examples](../examples.md) for complete notebook workflows.
