# Core Concepts

## Model steps describe the task

Methods charged with `TRAIN_STEP`, `VALIDATION_STEP`, `TEST_STEP`, and `PREDICT_STEP`
belong to the model. They receive an `EventContext`, usually read its `batch`, and
return task-specific values. Only define the workflows the application needs.

Train, validation, and test steps should return `StepOutput`. Its predictions and
targets let configured metrics reuse the same forward pass as the loss. A bare loss
or `(loss, metrics)` remains supported only when no automatic metrics are configured.

## Battery owns orchestration

`Battery` owns workflow mechanics:

1. Select and apply the device.
2. Dispatch before/after lifecycle events.
3. Set train or evaluation mode.
4. Run the charged model step.
5. Perform gradient operations during training.
6. Aggregate loss and metrics.
7. Return typed result dictionaries.

Public epoch numbers begin at one. Test and prediction are single-pass workflows and
therefore expose epoch `1` to handlers.

## Metrics describe measurement

Ordinary callables produce a value per batch. Stateful metrics implement
`reset`/`update`/`compute` for exact phase-level aggregation. `CollectedMetric` adapts
an ordinary callable by retaining detached CPU predictions and targets for the full
phase; its memory use grows with the dataset.

## Callbacks extend mechanics

Callbacks react to the same events as model lifecycle handlers. Built-ins implement
early stopping, checkpointing, experiment tracking, gradient accumulation, clipping,
mixed precision, and scheduling. Their configured order is significant for the
optimization extension points.

Callbacks inheriting from `Callback` can save and restore state in full checkpoints.
Decorator-only callback objects remain usable but do not participate in checkpoint
state.

## Results are ordinary mappings

`train` returns per-epoch loss and metric histories. `test` returns aggregate loss
and metrics. `predict` returns the model-defined output either per batch or recursively
concatenated. These mappings can be serialized or passed to plotting code without a
framework-specific history object.

## Events expose context, not hidden state

Every handler receives an `EventContext`. Available keys depend on the event; consult
the [Events API](../reference/events.md) before accessing a key. Context data should
be treated as scoped to the current event unless the event explicitly supports a
provider or executor result.
