# Training and Evaluation

## Define only the workflows you use

`Battery.train` and `Battery.fit` require `Event.TRAIN_STEP` and an optimizer.
`fit` accepts optional validation data and additionally requires
`Event.VALIDATION_STEP` when that data is available. Standalone `Battery.validate`,
testing, and prediction do not require an optimizer; each requires its corresponding
charged method.

```python
@charge(Event.TRAIN_STEP)
def training_step(self, context: EventContext) -> StepOutput:
    inputs, targets = context["batch"]
    predictions = self(inputs)
    return StepOutput(
        loss=F.cross_entropy(predictions, targets),
        predictions=predictions,
        targets=targets,
    )
```

The loss must be a scalar `torch.Tensor`. During training, its original value is
reported while optimization callbacks may divide the tensor used for backward.

## Step-result forms

`StepOutput` is the recommended form:

```python
return StepOutput(
    loss=loss,
    predictions=predictions,
    targets=targets,
    metrics={"mean_confidence": confidence},
)
```

Predictions and targets are required when `Battery(metrics=...)` is configured.
Manual metric values must be numeric scalars and override automatic metrics with the
same name.

For compatibility, a step may return either form below only when automatic metrics
are not configured:

```python
return loss
```

```python
return loss, {"accuracy": accuracy}
```

Invalid tuple shapes, non-dictionary metric payloads, non-scalar losses, and
non-numeric metrics fail immediately.

## Fit with optional validation

```python
history = battery.fit(
    train_loader,
    val_loader,
    epochs=20,
    verbose=1,
)
```

Loaders must implement `len()` and contain at least one batch. Validation runs after
each completed train epoch. Public epochs begin at one in all event contexts.

Without validation:

```python
history = battery.fit(train_loader, epochs=20)
assert history["val_loss"] == []
assert history["val_metrics"] == {}
```

`fit()` returns a `FitResult`. It does not fail when validation data is absent.

## Train without validation

Use `train()` for an intentionally training-only workflow:

```python
history = battery.train(train_loader, epochs=20)
```

For compatibility in 0.11.0, `train(..., val_loader=...)` and implicit DataPack
validation still run validation and populate `TrainResult.val_loss` and
`TrainResult.val_metrics`. That parameter and those fields are deprecated; the call
logs a warning and emits `DeprecationWarning` when validation actually runs. Migrate
combined workflows to `fit()`.

## Validate once

```python
validation_result = battery.validate(val_loader, verbose=0)
print(validation_result["val_loss"])
print(validation_result.get("val_metrics", {}))
```

Standalone validation runs one evaluation-only pass at epoch one with gradients
disabled. An explicit loader or validation data from the DataPack `"fit"` stage is
required.

## Evaluate once

```python
result = battery.test(test_loader, verbose=0)
print(result["test_loss"])
print(result.get("test_metrics", {}))
```

Validation and testing use evaluation mode and disable gradient tracking. `Battery`
does not restore the previous model mode afterward; a later training phase sets train
mode again.

## Result histories

Fitting returns an ordinary `FitResult` mapping:

```python
{
    "train_loss": [0.72, 0.51],
    "val_loss": [0.68, 0.47],
    "train_metrics": {"accuracy": [0.74, 0.82]},
    "val_metrics": {"accuracy": [0.76, 0.84]},
}
```

Loss and ordinary callable metrics are weighted by inferred batch size. Stateful
metrics supply their own phase aggregation. See [Metrics](metrics.md) before using a
non-decomposable measurement such as macro F1 or AUROC.

## Input validation

Training and fitting validate the complete configuration before dispatching lifecycle
events:

- `epochs` must be positive.
- The train loader must be sized and non-empty.
- An optimizer and train-step handler must exist.
- If validation is requested, its loader and handler must exist.
- `verbose` must be `0`, `1`, or `2`.

Exceptions raised by user steps or callbacks propagate. Active progress output is
aborted first so a failed run does not leave an open progress bar.
