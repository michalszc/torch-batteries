# Events API

Handlers receive an `EventContext`; keys vary by event. Epoch values are one-based.
Broadcast handlers run model-first and then in callback order. Provider/executor
events are exclusive and reject multiple handlers.

## Workflow events

| Phase | Once around workflow | Around each epoch | Around each batch | Required step |
| --- | --- | --- | --- | --- |
| Train | `BEFORE_TRAIN`, `AFTER_TRAIN` | `BEFORE_TRAIN_EPOCH`, `AFTER_TRAIN_EPOCH` | `BEFORE_TRAIN_STEP`, `AFTER_TRAIN_STEP` | `TRAIN_STEP` |
| Validation | `BEFORE_VALIDATION`, `AFTER_VALIDATION` | `BEFORE_VALIDATION_EPOCH`, `AFTER_VALIDATION_EPOCH` | `BEFORE_VALIDATION_STEP`, `AFTER_VALIDATION_STEP` | `VALIDATION_STEP` |
| Test | `BEFORE_TEST`, `AFTER_TEST` | `BEFORE_TEST_EPOCH`, `AFTER_TEST_EPOCH` | `BEFORE_TEST_STEP`, `AFTER_TEST_STEP` | `TEST_STEP` |
| Predict | `BEFORE_PREDICT`, `AFTER_PREDICT` | `BEFORE_PREDICT_EPOCH`, `AFTER_PREDICT_EPOCH` | `BEFORE_PREDICT_STEP`, `AFTER_PREDICT_STEP` | `PREDICT_STEP` |

## Optimization extension events

| Event | Dispatch | Purpose |
| --- | --- | --- |
| `SETUP` | Broadcast | Configure callbacks for the selected device |
| `STEP_EXECUTION_CONTEXT` | Context providers | Wrap model steps, for example with autocast |
| `CONFIGURE_TRAIN_STEP` | Exclusive provider | Select zeroing, loss division, and optimizer boundary |
| `BEFORE_BACKWARD` / `AFTER_BACKWARD` | Broadcast | Observe or adjust backward preparation/completion |
| `BACKWARD` | Exclusive executor | Replace ordinary `loss.backward()` |
| `BEFORE_GRADIENT_CLIP` | Broadcast | Prepare gradients, including AMP unscaling |
| `GRADIENT_CLIP` | Exclusive executor | Apply configured clipping |
| `BEFORE_OPTIMIZER_STEP` / `AFTER_OPTIMIZER_STEP` | Broadcast | Observe actual optimizer boundaries |
| `OPTIMIZER_STEP` | Exclusive executor | Replace ordinary `optimizer.step()` |

The generated `Event` reference below documents the exact context available for each
event and its default behavior.

::: torch_batteries.events
