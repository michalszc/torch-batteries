"""Tests for mixed precision control."""

from typing import cast

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from torch_batteries import Battery, Event, EventContext, StepOutput, charge
from torch_batteries.callbacks import MixedPrecision


class _AutocastModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(2, 1)
        self.autocast_states: list[bool] = []

    def _step(self, context: EventContext) -> StepOutput:
        inputs, targets = cast("tuple[torch.Tensor, torch.Tensor]", context["batch"])
        self.autocast_states.append(torch.is_autocast_enabled("cpu"))
        predictions = self.layer(inputs)
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

    @charge(Event.TEST_STEP)
    def test_step(self, context: EventContext) -> StepOutput:
        return self._step(context)

    @charge(Event.PREDICT_STEP)
    def predict_step(self, context: EventContext) -> torch.Tensor:
        inputs, _ = cast("tuple[torch.Tensor, torch.Tensor]", context["batch"])
        self.autocast_states.append(torch.is_autocast_enabled("cpu"))
        return cast("torch.Tensor", self.layer(inputs))


def _loader() -> DataLoader:
    return DataLoader(
        TensorDataset(torch.ones(2, 2), torch.zeros(2, 1)),
        batch_size=2,
    )


def test_amp_selects_cpu_bfloat16_for_all_phases() -> None:
    model = _AutocastModel()
    precision = MixedPrecision("amp")
    battery = Battery(
        model,
        device="cpu",
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        callbacks=[precision],
    )

    battery.train(_loader(), _loader(), verbose=0)
    battery.test(_loader(), verbose=0)
    battery.predict(_loader(), verbose=0)

    assert precision.effective_precision == "bf16-mixed"
    assert model.autocast_states == [True, True, True, True]


def test_full_precision_disables_autocast() -> None:
    model = _AutocastModel()
    battery = Battery(
        model,
        device="cpu",
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        callbacks=[MixedPrecision("32-true")],
    )

    battery.train(_loader(), verbose=0)

    assert model.autocast_states == [False]


def test_rejects_duplicate_mixed_precision_controls() -> None:
    model = _AutocastModel()

    with pytest.raises(ValueError, match="Only one MixedPrecision"):
        Battery(
            model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            callbacks=[MixedPrecision("amp"), MixedPrecision("32-true")],
        )


def test_state_round_trip_validates_configuration() -> None:
    control = MixedPrecision("amp")
    control.configure(torch.device("cpu"))
    state = control.state_dict()
    restored = MixedPrecision("amp")
    restored.configure(torch.device("cpu"))

    restored.load_state_dict(state)

    assert restored.effective_precision == "bf16-mixed"


def test_rejects_invalid_precision_and_device() -> None:
    with pytest.raises(ValueError, match="precision must be"):
        MixedPrecision("invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="does not support"):
        MixedPrecision("amp").configure(torch.device("meta"))


def test_rejects_invalid_or_incompatible_checkpoint_state() -> None:
    control = MixedPrecision("amp")
    control.configure(torch.device("cpu"))

    with pytest.raises(ValueError, match="Invalid MixedPrecision"):
        control.load_state_dict({})
    with pytest.raises(ValueError, match="does not match"):
        control.load_state_dict(
            {
                "precision": "32-true",
                "effective_precision": "32-true",
                "scaler": {},
            }
        )
    with pytest.raises(TypeError, match="scaler checkpoint"):
        control.load_state_dict(
            {
                "precision": "amp",
                "effective_precision": "bf16-mixed",
                "scaler": None,
            }
        )


def test_fp16_control_scales_unscales_and_steps_on_cpu() -> None:
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    control = MixedPrecision("16-mixed")
    control.configure(torch.device("cpu"))
    loss = model(torch.ones(1, 1)).sum()

    control.backward(loss)
    control.unscale_(optimizer)
    control.optimizer_step(optimizer)

    assert control.scaler.is_enabled()
