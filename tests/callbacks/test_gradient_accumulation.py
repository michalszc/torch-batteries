"""Integration tests for gradient accumulation."""

from typing import cast

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from torch_batteries import Battery, Event, EventContext, charge
from torch_batteries.callbacks import GradientAccumulation


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(1, 1, bias=False)
        nn.init.constant_(self.layer.weight, 1.0)

    @charge(Event.TRAIN_STEP)
    def training_step(self, context: EventContext) -> torch.Tensor:
        inputs, targets = cast("tuple[torch.Tensor, torch.Tensor]", context["batch"])
        return cast("torch.Tensor", ((self.layer(inputs) - targets) ** 2).mean())


class _StepRecorder:
    def __init__(self) -> None:
        self.records: list[tuple[bool, int]] = []

    @charge(Event.AFTER_TRAIN_STEP)
    def after_step(self, context: EventContext) -> None:
        self.records.append((context["optimizer_step"], context["optimizer_step_idx"]))


def _loader(samples: int) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.ones(samples, 1), torch.zeros(samples, 1)),
        batch_size=1,
        shuffle=False,
    )


def test_accumulates_and_steps_final_partial_group() -> None:
    model = _Model()
    recorder = _StepRecorder()
    control = GradientAccumulation(steps=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    battery = Battery(model, optimizer=optimizer, callbacks=[control, recorder])

    battery.train(_loader(5), epochs=1, verbose=0)

    assert recorder.records == [
        (False, 0),
        (True, 1),
        (False, 1),
        (True, 2),
        (True, 3),
    ]
    assert control.optimizer_step_idx == 3


def test_partial_group_uses_actual_group_size() -> None:
    accumulated_model = _Model()
    reference_model = _Model()
    accumulated = Battery(
        accumulated_model,
        optimizer=torch.optim.SGD(accumulated_model.parameters(), lr=0.1),
        callbacks=[GradientAccumulation(steps=4)],
    )
    reference = Battery(
        reference_model,
        optimizer=torch.optim.SGD(reference_model.parameters(), lr=0.1),
        callbacks=[GradientAccumulation(steps=2)],
    )

    accumulated.train(_loader(2), verbose=0)
    reference.train(_loader(2), verbose=0)

    assert torch.equal(accumulated_model.layer.weight, reference_model.layer.weight)


def test_rejects_duplicate_controls() -> None:
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match="Only one GradientAccumulation"):
        Battery(
            model,
            optimizer=optimizer,
            callbacks=[GradientAccumulation(2), GradientAccumulation(3)],
        )


def test_state_round_trip_and_configuration_validation() -> None:
    control = GradientAccumulation(steps=3)
    control.record_optimizer_step()
    state = control.state_dict()
    restored = GradientAccumulation(steps=3)

    restored.load_state_dict(state)

    assert restored.optimizer_step_idx == 1


def test_rejects_invalid_configuration_and_checkpoint_state() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        GradientAccumulation(0)

    control = GradientAccumulation(steps=2)
    with pytest.raises(ValueError, match="Invalid GradientAccumulation"):
        control.load_state_dict({})
    with pytest.raises(ValueError, match="do not match"):
        control.load_state_dict({"steps": 3, "optimizer_step_idx": 1})
