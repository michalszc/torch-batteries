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

## Predict named datasets

A DataPack can provide a mapping of prediction datasets:

```python
return DatasetBundle(
    predict={"Predict1": predict_1, "Predict2": predict_2},
)

results = battery.predict(move_to_cpu=True, concatenate=True)
predict_1_outputs = results["Predict1"]["predictions"]
```

A bare prediction dataset returns the ordinary `PredictResult` shape. A named mapping
always returns results keyed by dataset name, including when the mapping contains one
entry. Pass `dataset="Predict1"` to run only one and receive the singular result shape.
Dataset selection is available only for implicit DataPack workflows and cannot
accompany an explicit DataLoader.

## Stream large predictions

```python
for batch_output in battery.predict_iter(loader, move_to_cpu=True):
    write_batch(batch_output)
```

Streaming an implicit DataPack with multiple prediction datasets requires an explicit
selection, because otherwise each output's source would be ambiguous:

```python
for batch_output in battery.predict_iter(dataset="Predict1", move_to_cpu=True):
    write_batch(batch_output)
```

Streaming does not retain prior outputs in `Battery`. Fully consume the iterator for
`AFTER_PREDICT_EPOCH` and `AFTER_PREDICT` handlers to run. If an implicit DataPack
stream may stop early, explicitly close the generator so `TEARDOWN_DATA` runs
immediately:

```python
from contextlib import closing

with closing(
    battery.predict_iter(dataset="Predict1", move_to_cpu=True)
) as predictions:
    for batch_output in predictions:
        write_batch(batch_output)
        if finished:
            break
```

Closing an incomplete stream does not emit the successful-completion prediction
events. An ordinary `break` does not guarantee immediate cleanup when the iterator is
retained, so do not rely on garbage collection to release DataPack resources.

Prediction loaders must be sized and non-empty. A dataset may return features only;
the prediction step controls how its batch is unpacked and does not require targets.
