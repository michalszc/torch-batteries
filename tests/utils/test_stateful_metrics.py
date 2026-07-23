"""Tests for full-phase metric support."""

from typing import cast

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from torch_batteries import (
    Battery,
    CollectedMetric,
    Event,
    EventContext,
    StatefulMetric,
    StepOutput,
    charge,
)


class _PhaseMean:
    def __init__(self) -> None:
        self.total = 0.0
        self.samples = 0
        self.resets = 0

    def reset(self) -> None:
        self.total = 0.0
        self.samples = 0
        self.resets += 1

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        del targets
        self.total += float(predictions.sum())
        self.samples += predictions.numel()

    def compute(self) -> float:
        return self.total / self.samples


class _MetricModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def _step(self, context: EventContext) -> StepOutput:
        inputs, targets = cast("tuple[torch.Tensor, torch.Tensor]", context["batch"])
        predictions = inputs * self.scale
        return StepOutput(
            loss=((predictions - targets) ** 2).mean(),
            predictions=predictions,
            targets=targets,
        )

    @charge(Event.TRAIN_STEP)
    def training_step(self, context: EventContext) -> StepOutput:
        return self._step(context)

    @charge(Event.VALIDATION_STEP)
    def validation_step(self, context: EventContext) -> StepOutput:
        return self._step(context)


def test_stateful_metric_protocol_and_phase_reset() -> None:
    metric = _PhaseMean()
    assert isinstance(metric, StatefulMetric)
    model = _MetricModel()
    battery = Battery(
        model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        metrics={"mean": metric},
    )
    loader = DataLoader(
        TensorDataset(torch.tensor([[1.0], [2.0], [9.0]]), torch.zeros(3, 1)),
        batch_size=2,
    )

    result = battery.train(loader, loader, verbose=0)

    assert result["train_metrics"]["mean"] == [4.0]
    assert result["val_metrics"]["mean"] == [4.0]
    assert metric.resets == 2


def test_collected_metrics_receive_one_shared_full_phase() -> None:
    observed_shapes: list[torch.Size] = []

    def phase_range(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        del targets
        observed_shapes.append(predictions.shape)
        return predictions.max() - predictions.min()

    model = _MetricModel()
    battery = Battery(
        model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        metrics={
            "range": CollectedMetric(phase_range),
            "double_range": CollectedMetric(
                lambda predictions, targets: 2 * phase_range(predictions, targets)
            ),
        },
    )
    loader = DataLoader(
        TensorDataset(torch.tensor([[1.0], [2.0], [9.0]]), torch.zeros(3, 1)),
        batch_size=2,
    )

    result = battery.train(loader, verbose=0)

    assert result["train_metrics"]["range"] == [8.0]
    assert result["train_metrics"]["double_range"] == [16.0]
    assert observed_shapes == [torch.Size([3, 1]), torch.Size([3, 1])]
