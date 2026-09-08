"""Tests for Battery configuration mutation boundaries."""

from typing import Literal, cast

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from torch_batteries import (
    Battery,
    DataContext,
    DataLoaderConfig,
    DataPack,
    DatasetBundle,
    Event,
    EventContext,
    charge,
)


class _TrainModel(nn.Module):
    def __init__(self, *, mutate_during_step: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.mutate_during_step = mutate_during_step

    @charge(Event.TRAIN_STEP)
    def training_step(self, context: EventContext) -> torch.Tensor:
        if self.mutate_during_step:
            context["battery"].optimizer = None
        inputs, targets = cast("tuple[torch.Tensor, torch.Tensor]", context["batch"])
        return ((inputs * self.weight - targets) ** 2).mean()


class _CallbackMutator:
    def __init__(self, setting: Literal["optimizer", "metrics", "policy"]) -> None:
        self.setting = setting

    @charge(Event.BEFORE_TRAIN)
    def mutate(self, context: EventContext) -> None:
        battery = context["battery"]
        if self.setting == "optimizer":
            battery.optimizer = None
        elif self.setting == "metrics":
            battery.metrics = {}
        else:
            battery.metric_error_policy = "warn"


class _DataPackMutator(DataPack):
    @charge(Event.SETUP_DATA)
    def setup(self, context: DataContext) -> DatasetBundle:
        context["battery"].metrics = {}
        dataset = TensorDataset(torch.ones(1, 1), torch.zeros(1, 1))
        return DatasetBundle(train=dataset)

    @charge(Event.CONFIGURE_DATALOADER)
    def configure_loader(self, _: DataContext) -> DataLoaderConfig:
        return DataLoaderConfig(batch_size=1)


def _loader() -> DataLoader:
    return DataLoader(
        TensorDataset(torch.ones(1, 1), torch.zeros(1, 1)),
        batch_size=1,
    )


@pytest.mark.parametrize("setting", ["optimizer", "metrics", "policy"])
def test_callbacks_cannot_replace_battery_configuration(setting: str) -> None:
    model = _TrainModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    callback = _CallbackMutator(
        cast("Literal['optimizer', 'metrics', 'policy']", setting)
    )
    battery = Battery(model, optimizer=optimizer, callbacks=[callback])

    with pytest.raises(RuntimeError, match="while an event handler is running"):
        battery.train(_loader(), verbose=0)

    battery.optimizer = optimizer
    battery.metrics = {}
    battery.metric_error_policy = "raise"


def test_model_step_cannot_replace_battery_configuration() -> None:
    model = _TrainModel(mutate_during_step=True)
    battery = Battery(
        model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
    )

    with pytest.raises(RuntimeError, match=r"Battery\.optimizer"):
        battery.train(_loader(), verbose=0)


def test_data_pack_event_cannot_replace_battery_configuration() -> None:
    model = _TrainModel()
    battery = Battery(
        model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        data_pack=_DataPackMutator(),
    )

    with pytest.raises(RuntimeError, match=r"Battery\.metrics"):
        battery.train(verbose=0)
