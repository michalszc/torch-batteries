"""Tests for gradient clipping control."""

from typing import Any, cast

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from torch_batteries import Battery, Event, EventContext, charge
from torch_batteries.callbacks import GradientAccumulation, GradientClip


class _GradientModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[10.0]]))

    @charge(Event.TRAIN_STEP)
    def training_step(self, context: EventContext) -> torch.Tensor:
        inputs, targets = cast("tuple[torch.Tensor, torch.Tensor]", context["batch"])
        return ((inputs @ self.weight - targets) ** 2).mean()


def _loader(samples: int = 1) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.ones(samples, 1), torch.zeros(samples, 1)),
        batch_size=1,
    )


def test_norm_clipping_limits_combined_gradient() -> None:
    parameter = nn.Parameter(torch.tensor([1.0, 2.0]))
    parameter.grad = torch.tensor([3.0, 4.0])

    pre_clip = GradientClip(1.0, "norm").apply([parameter])

    assert pre_clip == pytest.approx(5.0)
    assert parameter.grad.norm().item() == pytest.approx(1.0)


def test_value_clipping_clamps_each_gradient_element() -> None:
    parameter = nn.Parameter(torch.tensor([1.0, 2.0]))
    parameter.grad = torch.tensor([-3.0, 4.0])

    GradientClip(0.5, "value").apply([parameter])

    assert torch.equal(parameter.grad, torch.tensor([-0.5, 0.5]))


def test_clipping_runs_only_at_accumulated_optimizer_boundary(
    mocker: Any,
) -> None:
    model = _GradientModel()
    clipping = GradientClip(1.0)
    apply_spy = mocker.spy(clipping, "apply")
    battery = Battery(
        model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        callbacks=[GradientAccumulation(steps=2), clipping],
    )

    battery.train(_loader(3), verbose=0)

    assert apply_spy.call_count == 2


def test_rejects_invalid_and_duplicate_configuration() -> None:
    with pytest.raises(ValueError, match="greater than or equal"):
        GradientClip(-1)
    with pytest.raises(ValueError, match="'norm' or 'value'"):
        GradientClip(1, "unsupported")  # type: ignore[arg-type]

    model = _GradientModel()
    with pytest.raises(ValueError, match="accepts exactly one handler"):
        Battery(
            model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            callbacks=[GradientClip(1), GradientClip(2)],
        )


def test_checkpoint_configuration_is_validated() -> None:
    clipping = GradientClip(1.0, "norm")

    clipping.load_state_dict(clipping.state_dict())
    with pytest.raises(ValueError, match="does not match"):
        clipping.load_state_dict({"value": 2.0, "algorithm": "value"})
