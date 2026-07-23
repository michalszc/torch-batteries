"""Tests for learning-rate scheduler callbacks."""

from typing import cast

import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, TensorDataset

from torch_batteries import Battery, Event, EventContext, charge
from torch_batteries.callbacks import GradientAccumulation, LearningRateScheduler


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(1, 1)

    @charge(Event.TRAIN_STEP)
    def training_step(self, context: EventContext) -> torch.Tensor:
        inputs, targets = cast("tuple[torch.Tensor, torch.Tensor]", context["batch"])
        return cast("torch.Tensor", ((self.layer(inputs) - targets) ** 2).mean())


def _loader(samples: int = 4) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.ones(samples, 1), torch.zeros(samples, 1)),
        batch_size=1,
    )


def test_step_scheduler_advances_only_on_optimizer_steps() -> None:
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    battery = Battery(
        model,
        optimizer=optimizer,
        callbacks=[
            GradientAccumulation(steps=2),
            LearningRateScheduler(scheduler, interval="step"),
        ],
    )

    battery.train(_loader(), verbose=0)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.025)


def test_epoch_scheduler_advances_once_per_epoch() -> None:
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    battery = Battery(
        model,
        optimizer=optimizer,
        callbacks=[LearningRateScheduler(scheduler)],
    )

    battery.train(_loader(1), epochs=2, verbose=0)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.025)


def test_plateau_scheduler_uses_selected_validation_metric() -> None:
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=0, factor=0.5)
    callback = LearningRateScheduler(
        scheduler,
        stage="val",
        metric="loss",
    )

    callback.on_validation_end(EventContext(epoch=0, val_metrics={"loss": 1.0}))
    callback.on_validation_end(EventContext(epoch=1, val_metrics={"loss": 1.0}))

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)


def test_validates_plateau_and_ordinary_scheduler_configuration() -> None:
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match="requires interval='epoch'"):
        LearningRateScheduler(ReduceLROnPlateau(optimizer), interval="step")
    with pytest.raises(ValueError, match="only supported"):
        LearningRateScheduler(StepLR(optimizer, 1), stage="val", metric="loss")


def test_scheduler_state_round_trip() -> None:
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    callback = LearningRateScheduler(StepLR(optimizer, 1))
    optimizer.step()
    callback.on_train_epoch_end(EventContext(epoch=0, train_metrics={"loss": 1.0}))
    state = callback.state_dict()

    restored_model = _Model()
    restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.1)
    restored = LearningRateScheduler(StepLR(restored_optimizer, 1))
    restored.load_state_dict(state)

    assert restored.scheduler.last_epoch == callback.scheduler.last_epoch
