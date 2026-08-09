# Batches and Devices

`Battery` moves each loader batch to its selected device before dispatching step
events. The step reads the moved value from `context["batch"]`.

## Tuple batches

PyTorch's common supervised format works directly:

```python
inputs, targets = context["batch"]
```

Batch size is inferred from the first tensor in the tuple or list.

## Dictionary batches

Dictionary keys are preserved and tensor values are moved recursively:

```python
@charge(Event.TRAIN_STEP)
def training_step(self, context: EventContext) -> StepOutput:
    batch = context["batch"]
    predictions = self(batch["image"], batch["features"])
    return StepOutput(
        loss=F.cross_entropy(predictions, batch["label"]),
        predictions=predictions,
        targets=batch["label"],
    )
```

Metric inputs come from `StepOutput`, so automatic metrics do not assume positional
dictionary keys.

## Nested and multiple inputs

Recursive movement supports tensors nested inside:

- Lists
- Tuples and named tuples
- Dictionaries

Non-tensor leaves such as strings, paths, and sample identifiers remain unchanged.
Dataclass instances and custom objects are also left unchanged; convert them to a
supported container inside the dataset/collate function when their tensors need
automatic placement.

Batch-size inference supports a tensor directly or the first tensor at the top level
of a list, tuple, or dictionary. It does not recursively search nested containers.
When no such tensor exists, the fallback size is one, which affects loss and
callable-metric weighting.

## Automatic and explicit devices

```python
battery = Battery(model, optimizer=optimizer)  # CUDA, then MPS, then CPU
battery = Battery(model, optimizer=optimizer, device="cpu")
battery = Battery(model, optimizer=optimizer, device="cuda:1")
```

The model is moved during construction. If full optimizer state is later loaded from
a checkpoint, its tensors are recursively moved to the same device.

## Prediction output devices

Prediction outputs remain on the model device by default. Use:

```python
result = battery.predict(loader, move_to_cpu=True)
```

or:

```python
for batch_output in battery.predict_iter(loader, move_to_cpu=True):
    consume(batch_output)
```

CPU transfer recursively detaches tensor leaves while preserving matching containers.
