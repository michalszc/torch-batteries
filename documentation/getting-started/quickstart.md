# Quick Start

This complete CPU example trains and validates a regression model, evaluates it,
and produces concatenated CPU predictions.

```python
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from torch_batteries import Battery, Event, EventContext, StepOutput, charge


class Regressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)

    def supervised_step(self, context: EventContext) -> StepOutput:
        inputs, targets = context["batch"]
        predictions = self(inputs)
        return StepOutput(
            loss=F.mse_loss(predictions, targets),
            predictions=predictions,
            targets=targets,
        )

    @charge(Event.TRAIN_STEP)
    def training_step(self, context: EventContext) -> StepOutput:
        return self.supervised_step(context)

    @charge(Event.VALIDATION_STEP)
    def validation_step(self, context: EventContext) -> StepOutput:
        return self.supervised_step(context)

    @charge(Event.TEST_STEP)
    def test_step(self, context: EventContext) -> StepOutput:
        return self.supervised_step(context)

    @charge(Event.PREDICT_STEP)
    def predict_step(self, context: EventContext) -> torch.Tensor:
        inputs = context["batch"][0]
        return self(inputs)


torch.manual_seed(7)
inputs = torch.randn(96, 4)
targets = inputs.sum(dim=1, keepdim=True)
train_data, val_data, test_data = torch.utils.data.random_split(
    TensorDataset(inputs, targets),
    [64, 16, 16],
    generator=torch.Generator().manual_seed(7),
)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16)
test_loader = DataLoader(test_data, batch_size=16)

model = Regressor()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
battery = Battery(
    model,
    device="cpu",
    optimizer=optimizer,
    metrics={"mae": lambda pred, target: F.l1_loss(pred, target)},
)

history = battery.train(train_loader, val_loader, epochs=3, verbose=0)
test_result = battery.test(test_loader, verbose=0)
prediction_result = battery.predict(
    test_loader,
    verbose=0,
    move_to_cpu=True,
    concatenate=True,
)

print(history["train_loss"])
print(history["val_metrics"]["mae"])
print(test_result)
print(prediction_result["predictions"].shape)
```

## What `Battery` handled

The model defined the task-specific forward pass and loss. `Battery` selected a
device, moved every batch, set train/evaluation modes, ran backward and optimizer
steps, calculated metrics from the same predictions, aggregated results, and emitted
lifecycle events.

The result uses one list entry per completed epoch. Testing returns one aggregate.
Prediction normally returns a list of batch outputs; `concatenate=True` produced one
tensor here.

Continue with [Core Concepts](core-concepts.md), then select a task from the
[Guides](../guides/index.md).

This example passes DataLoaders directly, which is useful when the caller already
owns them. For reusable event-driven dataset and loader construction, continue with
the [DataPack guide](../guides/data-pack.md).
