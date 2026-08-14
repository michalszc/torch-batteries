# Callbacks and Optimization

Callbacks are ordinary objects with charged methods. They run after matching model
lifecycle handlers and in the order supplied to `Battery`. Inheriting from `Callback`
adds resumable `state_dict`/`load_state_dict` support; decorator-only objects remain
valid for stateless behavior.

```python
battery = Battery(
    model,
    optimizer=optimizer,
    callbacks=[accumulation, precision, clipping, scheduler],
)
```

## Early stopping

```python
early_stopping = EarlyStopping(
    phase="val",
    metric="loss",
    mode="min",
    patience=5,
    min_delta=1e-4,
    restore_best_weights=True,
)
```

`phase` is `"train"` or `"val"`. The monitored metric can be `"loss"` or a metric
produced by the phase. Patience counts consecutive completed monitored phases without
the required improvement. Best weights are cloned to CPU-safe independent tensors
and restored after training when requested. Full checkpoints preserve the best score,
patience counter, and optional best weights.

For compatibility, the callbacks still accept `stage=` as a deprecated keyword
alias. New code should use `phase=`.

## Gradient accumulation

```python
accumulation = GradientAccumulation(steps=4)
```

Loss is divided by the actual accumulation-group size before backward. Gradients are
zeroed at group start and the optimizer advances at group end. A short final group is
normalized by its own size. Optimizer-step counters, step schedulers, mixed precision,
and clipping observe real optimizer boundaries rather than every loader batch.

## Gradient clipping

```python
clip_norm = GradientClip(value=1.0, algorithm="norm")
clip_value = GradientClip(value=0.5, algorithm="value")
```

Clipping runs immediately before a real optimizer step. Norm clipping returns the
pre-clip norm internally; value clipping clamps each gradient element. With mixed
precision, gradients are unscaled before clipping.

## Mixed precision

```python
precision = MixedPrecision("amp")
```

Supported modes are:

| Mode | Behavior |
| --- | --- |
| `32-true` | Disable autocast and scaling |
| `16-mixed` | FP16 autocast with gradient scaling |
| `bf16-mixed` | BF16 autocast without FP16 scaling |
| `amp` | BF16 on CPU, FP16 on CUDA/MPS |

The callback wraps train, validation, test, and prediction steps. Its scaler state is
included in a full checkpoint.

## Learning-rate scheduling

Advance an ordinary scheduler after each optimizer step:

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    total_steps=100,
)
callback = LearningRateScheduler(scheduler, interval="step")
```

Advance an ordinary scheduler after each train epoch:

```python
callback = LearningRateScheduler(scheduler, interval="epoch")
```

`ReduceLROnPlateau` requires an epoch interval, phase, and metric:

```python
callback = LearningRateScheduler(
    torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
    interval="epoch",
    phase="val",
    metric="loss",
)
```

A validation-monitored plateau scheduler requires a validation loader. Scheduler
configuration and advancement state are restored strictly from full checkpoints.

## Custom callbacks

```python
class EpochReporter(Callback):
    @charge(Event.AFTER_TRAIN_EPOCH)
    def report(self, context: EventContext) -> None:
        print(context["epoch"], context["train_metrics"])
```

Provider and executor events are exclusive: only one handler may own configuration,
backward, clipping, or optimizer execution. Conflicts fail during `Battery`
construction rather than producing order-dependent optimization.
