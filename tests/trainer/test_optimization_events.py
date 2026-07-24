"""Integration tests for model-defined optimization events."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from torch_batteries import (
    Battery,
    Event,
    EventContext,
    OptimizationStep,
    charge,
)


def _loader() -> DataLoader:
    return DataLoader(
        TensorDataset(torch.ones(1, 1), torch.zeros(1, 1)),
        batch_size=1,
    )


class _OptimizationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[1.0]]))
        self.events: list[str] = []

    @charge(Event.SETUP)
    def setup(self, context: EventContext) -> None:
        assert context["device"].type == "cpu"
        self.events.append("setup")

    @charge(Event.STEP_EXECUTION_CONTEXT)
    @contextmanager
    def execution_context(self, context: EventContext) -> Generator[None]:
        assert context["phase"] == "train"
        self.events.append("enter")
        try:
            yield
        finally:
            self.events.append("exit")

    @charge(Event.CONFIGURE_TRAIN_STEP)
    def configure_step(self, context: EventContext) -> OptimizationStep:
        assert context["total_batches"] == 1
        self.events.append("configure")
        return OptimizationStep(loss_divisor=2)

    @charge(Event.TRAIN_STEP)
    def training_step(self, context: EventContext) -> torch.Tensor:
        inputs, targets = cast("tuple[torch.Tensor, torch.Tensor]", context["batch"])
        self.events.append("train")
        return ((inputs @ self.weight - targets) ** 2).mean()

    @charge(Event.BACKWARD)
    def backward(self, context: EventContext) -> None:
        self.events.append("backward")
        context["backward_loss"].backward()

    @charge(Event.GRADIENT_CLIP)
    def gradient_clip(self, context: EventContext) -> None:
        del context
        self.events.append("clip")

    @charge(Event.OPTIMIZER_STEP)
    def optimizer_step(self, context: EventContext) -> None:
        self.events.append("step")
        optimizer = context["optimizer"]
        assert optimizer is not None
        optimizer.step()

    @charge(Event.AFTER_OPTIMIZER_STEP)
    def after_optimizer_step(self, context: EventContext) -> None:
        assert context["optimizer_step_idx"] == 1
        self.events.append("after-step")


def test_model_can_own_complete_optimization_lifecycle() -> None:
    model = _OptimizationModel()
    battery = Battery(
        model,
        device="cpu",
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
    )

    battery.train(_loader(), verbose=0)

    assert model.weight.item() == pytest.approx(0.9)
    assert model.events == [
        "setup",
        "configure",
        "enter",
        "train",
        "exit",
        "backward",
        "clip",
        "step",
        "after-step",
    ]


class _FailingOptimizerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[1.0]]))
        self.after_steps = 0

    @charge(Event.TRAIN_STEP)
    def training_step(self, context: EventContext) -> torch.Tensor:
        inputs, _ = cast("tuple[torch.Tensor, torch.Tensor]", context["batch"])
        return (inputs @ self.weight).sum()

    @charge(Event.OPTIMIZER_STEP)
    def optimizer_step(self, context: EventContext) -> None:
        del context
        raise RuntimeError

    @charge(Event.AFTER_OPTIMIZER_STEP)
    def after_optimizer_step(self, context: EventContext) -> None:
        del context
        self.after_steps += 1


def test_failed_optimizer_does_not_emit_success_or_increment_counter(
    tmp_path: Path,
) -> None:
    model = _FailingOptimizerModel()
    battery = Battery(
        model,
        device="cpu",
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
    )

    with pytest.raises(RuntimeError):
        battery.train(_loader(), verbose=0)

    checkpoint = tmp_path / "failed.pth"
    battery.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["optimizer_step_idx"] == 0
    assert model.after_steps == 0


class _InvalidPlanModel(_FailingOptimizerModel):
    @charge(Event.CONFIGURE_TRAIN_STEP)
    def configure_step(self, context: EventContext) -> object:
        del context
        return "invalid"


def test_invalid_model_optimization_plan_is_rejected() -> None:
    model = _InvalidPlanModel()
    battery = Battery(
        model,
        device="cpu",
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
    )

    with pytest.raises(TypeError, match="must return an OptimizationStep"):
        battery.train(_loader(), verbose=0)
