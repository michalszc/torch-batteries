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

The loss must be a scalar `torch.Tensor` and should normally be the mean loss for the
batch. Battery weights reported batch losses by the inferred batch size when it builds
phase and epoch results. If a step calculates a summed loss, normalize it in user code
before returning it. During training, the returned value is reported while optimization
callbacks may divide the tensor used for backward.

## Optionally compile the model

`torch.compile` is optional. When using it, compile the model first, then construct the
optimizer and `Battery` from the compiled model. This keeps charged-method discovery
and optimizer parameters attached to the same model object.

```python
model = torch.compile(MyModel())
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
battery = Battery(model, optimizer=optimizer)
```

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

A step may also return either form below when automatic metrics are not configured:

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
each completed train epoch by default. Pass `validate_every_n_epochs=5` to validate
on absolute epochs 5, 10, and so on, including after a checkpoint resume. Public
epochs begin at one in all event contexts.

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

`train()` runs training only. Use `fit()` when validation is needed, including when a
DataPack provides a validation dataset.

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
    "epochs_completed": 2,
    "optimizer_steps": 40,
    "stopped_early": False,
    "stop_reason": None,
}
```

Loss and ordinary callable metrics are weighted by inferred batch size. Stateful
metrics supply their own phase aggregation. See [Metrics](metrics.md) before using a
non-decomposable measurement such as macro F1 or AUROC.

The counters are cumulative across resume. `val_loss` and `val_metrics` contain entries
only for epochs when validation ran. A callback can call
`battery.request_stop("reason for stopping")`; the result then has
`stopped_early=True` and that `stop_reason`. Early stopping supplies a descriptive
reason automatically. Exceptions propagate and are not reported as a completed run.

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
