# Metrics

Metrics are named values aggregated during train, validation, and test phases. All
automatic metrics receive detached predictions and targets from `StepOutput`; no
second model forward pass is performed.

## Ordinary callables

Use a callable when averaging its per-batch value by sample count is mathematically
correct:

```python
def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    labels = predictions.argmax(dim=1)
    return (labels == targets).float().mean()


battery = Battery(
    model,
    optimizer=optimizer,
    metrics={"accuracy": accuracy},
)
```

The callable must return a Python numeric value or scalar tensor. A metric that raises
is logged and skipped for the remainder of that phase.

## Stateful phase metrics

Use `StatefulMetric` for measurements that must aggregate sufficient statistics over
the complete phase:

```python
class ExactAccuracy:
    def __init__(self) -> None:
        self.correct = 0
        self.total = 0

    def reset(self) -> None:
        self.correct = 0
        self.total = 0

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        self.correct += int((predictions.argmax(dim=1) == targets).sum())
        self.total += targets.numel()

    def compute(self) -> float:
        return self.correct / self.total
```

The object is reset before every phase, updated once per batch, and computed once at
the end. It may optionally implement `state_dict` and `load_state_dict` to participate
in full checkpoints.

## Full-phase callable metrics

`CollectedMetric` adapts a callable that needs all predictions and targets together:

```python
battery = Battery(
    model,
    optimizer=optimizer,
    metrics={"macro_f1": CollectedMetric(macro_f1)},
)
```

It retains detached CPU tensors and concatenates them at phase end. Memory use grows
with the dataset, so prefer an incremental stateful implementation for large outputs.
Multiple collected metrics share one retained prediction/target collection.

## Manual step metrics

A step can report task-specific scalar values directly:

```python
return StepOutput(
    loss=loss,
    predictions=predictions,
    targets=targets,
    metrics={"mean_confidence": predictions.softmax(dim=1).amax(dim=1).mean()},
)
```

Manual metrics are aggregated by batch size and override configured metrics with the
same name for that phase. Avoid using an ordinary batch average for non-decomposable
metrics.

## Names across the API

Given `metrics={"accuracy": metric}`, the same name appears in:

```python
history["train_metrics"]["accuracy"]
history["val_metrics"]["accuracy"]
test_result["test_metrics"]["accuracy"]
```

It is also the name used by early stopping, checkpoint monitoring, progress output,
and tracking callbacks. A misspelled monitored name raises for early stopping and is
ignored by model checkpointing, so keep one consistent vocabulary.
