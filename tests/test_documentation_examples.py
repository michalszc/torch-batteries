"""Executable contracts used by the Getting Started documentation."""

from typing import cast

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812
from torch.utils.data import DataLoader, TensorDataset

from torch_batteries import (
    Battery,
    DataContext,
    DataLoaderConfig,
    DataPack,
    DatasetBundle,
    Event,
    EventContext,
    StepOutput,
    charge,
)


class _DocumentedRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear.forward(inputs)

    def _supervised_step(self, context: EventContext) -> StepOutput:
        inputs, targets = cast("tuple[torch.Tensor, torch.Tensor]", context["batch"])
        predictions = self.forward(inputs)
        return StepOutput(
            loss=F.mse_loss(predictions, targets),
            predictions=predictions,
            targets=targets,
        )

    @charge(Event.TRAIN_STEP)
    def training_step(self, context: EventContext) -> StepOutput:
        return self._supervised_step(context)

    @charge(Event.VALIDATION_STEP)
    def validation_step(self, context: EventContext) -> StepOutput:
        return self._supervised_step(context)

    @charge(Event.TEST_STEP)
    def test_step(self, context: EventContext) -> StepOutput:
        return self._supervised_step(context)

    @charge(Event.PREDICT_STEP)
    def predict_step(self, context: EventContext) -> torch.Tensor:
        inputs, _ = cast("tuple[torch.Tensor, torch.Tensor]", context["batch"])
        return self.forward(inputs)


class _DocumentedDataPack(DataPack):
    seed = 7

    def __init__(self) -> None:
        self.teardown_calls = 0

    @charge(Event.SETUP_DATA)
    def setup(self, context: DataContext) -> DatasetBundle:
        inputs = torch.randn(16, 4, generator=context["generator"])
        targets = inputs.sum(dim=1, keepdim=True)
        dataset = TensorDataset(inputs, targets)
        return DatasetBundle(
            train=dataset,
            validation=dataset,
            test=dataset,
            predict=dataset,
        )

    @charge(Event.CONFIGURE_DATALOADER)
    def configure_loader(self, context: DataContext) -> DataLoaderConfig:
        return DataLoaderConfig(batch_size=8)

    @charge(Event.TEARDOWN_DATA)
    def teardown(self, context: DataContext) -> None:
        self.teardown_calls += 1


def test_getting_started_workflow() -> None:
    """The documented train/test/predict workflow remains executable on CPU."""
    torch.manual_seed(7)
    inputs = torch.randn(16, 4)
    targets = inputs.sum(dim=1, keepdim=True)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=8)
    model = _DocumentedRegressor()
    battery = Battery(
        model,
        device="cpu",
        optimizer=torch.optim.Adam(model.parameters(), lr=0.05),
        metrics={"mae": F.l1_loss},
    )

    history = battery.train(loader, loader, epochs=2, verbose=0)
    test_result = battery.test(loader, verbose=0)
    prediction_result = battery.predict(
        loader,
        verbose=0,
        move_to_cpu=True,
        concatenate=True,
    )

    assert len(history["train_loss"]) == 2
    assert len(history["val_metrics"]["mae"]) == 2
    assert "mae" in test_result["test_metrics"]
    assert prediction_result["predictions"].shape == (16, 1)
    assert prediction_result["predictions"].device.type == "cpu"


def test_documented_data_pack_workflow() -> None:
    """The documented implicit-loader workflow remains executable on CPU."""
    data_pack = _DocumentedDataPack()
    model = _DocumentedRegressor()
    battery = Battery(
        model,
        device="cpu",
        optimizer=torch.optim.Adam(model.parameters(), lr=0.05),
        data_pack=data_pack,
    )

    history = battery.train(epochs=1, verbose=0)
    test_result = battery.test(verbose=0)
    predictions = battery.predict(verbose=0, concatenate=True)

    assert len(history["train_loss"]) == 1
    assert test_result["test_loss"] >= 0
    assert predictions["predictions"].shape == (16, 1)
    assert data_pack.teardown_calls == 3
