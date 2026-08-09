# Prediction

Prediction steps return any user-defined structure. They run in evaluation mode with
gradient tracking disabled and receive batches already moved to the battery device.

```python
@charge(Event.PREDICT_STEP)
def predict_step(self, context: EventContext) -> dict[str, torch.Tensor]:
    inputs, sample_ids = context["batch"]
    logits = self(inputs)
    return {"sample_id": sample_ids, "probability": logits.softmax(dim=1)}
```

## Preserve batch outputs

```python
result = battery.predict(loader)
for batch_output in result["predictions"]:
    consume(batch_output)
```

This is the default and supports arbitrary outputs. Tensor leaves stay on their
current device and remain one entry per emitted batch.

## Move outputs to CPU

```python
result = battery.predict(loader, move_to_cpu=True)
```

Tensor leaves are detached and recursively moved while dictionaries, tuples, named
tuples, and lists retain their structure.

## Concatenate matching batches

```python
result = battery.predict(
    loader,
    move_to_cpu=True,
    concatenate=True,
)
probabilities = result["predictions"]["probability"]
```

Tensor leaves concatenate along dimension zero. Every batch must return the same
dictionary keys and matching container lengths; tensor shapes must be compatible
outside their first dimension. Unsupported leaves and mismatched structures raise.

## Stream large predictions

```python
for batch_output in battery.predict_iter(loader, move_to_cpu=True):
    write_batch(batch_output)
```

Streaming does not retain prior outputs in `Battery`. The iterator must be fully
consumed for `AFTER_PREDICT_EPOCH` and `AFTER_PREDICT` handlers to run. Breaking early
closes the generator without emitting those successful-completion events.

Prediction loaders must be sized and non-empty. A dataset may return features only;
the prediction step controls how its batch is unpacked and does not require targets.
